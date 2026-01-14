import json
import os
import re
import asyncio
import httpx
import logging

from datetime import datetime
from math import log2
from pprint import pformat
from typing import AsyncGenerator, Optional
from dotenv import load_dotenv

from pydantic import BaseModel
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from taxoenrich.core import RuWordNet
from utils import *
from io_utils import read_yaml
from tools import tools as available_tools_config
from reranker import Reranker, RerankerRequest
from interpreter import Interpreter, InterpreterRequest
from data_processing import prepare_target

load_dotenv()
config = read_yaml('config.yml')

app = FastAPI(title="RuWordNet Taxonomy Prediction API")

# Initialize RuWordNet
wordnet = RuWordNet('./wordnets/RuWordNet')

# Initialize OpenAI client
oclient = AsyncOpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Initialize logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Инициализация реранкера и интерпретатора
reranker = Reranker(oclient, os.environ['MODEL_NAME'], logger)
interpreter = Interpreter(oclient, os.environ['MODEL_NAME'], logger)


# Глобальная структура для отслеживания выбранных синсетов
class SynsetTracker:
    def __init__(self):
        self.selected_synsets = []
        self.target_word = None
    
    def add_synset(self, synset_id: str, function_name: str, args: dict):
        """Добавить выбранный синсет в трекер"""
        self.selected_synsets.append({
            'synset_id': synset_id,
            'function': function_name,
            'args': args,
            'timestamp': datetime.now().isoformat()
        })
    
    def set_target(self, target: str):
        """Установить целевое слово"""
        self.target_word = target

    
    def clear(self):
        """Очистить трекер"""
        self.selected_synsets.clear()
        self.target_word = None
    
    def get_tracking_data(self):
        """Получить данные отслеживания"""
        return {
            'target_word': self.target_word,
            'selected_synsets': self.selected_synsets,
            'total_selections': len(self.selected_synsets)
        }


# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    max_iterations: int = 50
    temperature: float = 0.5
    top_p: float = 0.95
    reranking: bool = False
    interpreting: bool = False
    functions: list = ["get_hyponyms"]
    output_file: str = None
    start_node_id: str = None  # Новый параметр для стартового узла

class PredictionResponse(BaseModel):
    result: str
    iterations: int
    full_conversation: list = None
    tracking_data: dict = None  # Данные отслеживания


def extract_target_word(text: str) -> str:
    """Извлечь целевое слово из тегов <predict_kb>"""
    match = re.search(r'<predict_kb>(.*?)</predict_kb>', text)
    return prepare_target(match.group(1).strip()) if match else "unknown"

def get_synset_id_from_response(text: str) -> str:
    """Извлечь id синсета из текста"""
    match = re.findall(r'\d{1,6}-[ANV]', text)
    return match[-1] if match else None

async def prepare_start_user_message(text: str, start_node_id: str = None, reranking: bool = False, sense: Optional[str] = None, tracker: Optional[SynsetTracker] = None) -> str:
    """Prepare user message with optional start node context"""
    if not start_node_id:
        return text
    
    # Получаем информацию о стартовом узле
    try:
        synset = wordnet.synsets.get(start_node_id)
        if not synset:
            logger.warning(f'Стартовый узел {start_node_id} не найден.')
            return text  # Fallback к оригинальному тексту
        
        synset_name = synset.synset_name
        synset_words = "; ".join(list(synset.synset_words)[:5])
        
        # Получаем гипонимы и гиперонимы
        hyponyms_info = await get_hyponyms(start_node_id, reranking=reranking, sense=sense, tracker=tracker)
        hypernyms_info = await get_hypernyms(start_node_id, reranking=False, sense=sense, tracker=tracker)
        
        # Формируем расширенное сообщение
        enhanced_message = f"""Дан стартовый узел для анализа:

**Стартовый синсет:** {synset_name} `{start_node_id}`
**Слова синсета:** {synset_words}

**Релевантные гипонимы**:
{hyponyms_info}

**Гиперонимы**:
{hypernyms_info}

**Контекст со словом для анализа**:

{text}

Начни анализ с этого узла и определи, куда лучше поместить целевое понятие. Ты можешь двигаться вниз по таксономии или вверх в зависимости от доступных функций. Если стартовый узел совсем не подходит по контексту, начни движение с корневых узлов."""
        
        return enhanced_message
        
    except Exception as e:
        logger.error(f"Ошибка подготовки сообщения со стартовым узлом: {e}")
        return text


async def get_hyponyms(node_id, reranking=False, sense: Optional[str] = None, context='', threshold=0, tracker: SynsetTracker = None):
    """Tool function for getting hyponyms and formatting as markdown"""
    if node_id == 'null':
        node_id = None
    
    results = wordnet.get_hyponyms(node_id, pos='N')

    # Format as clean markdown
    if not results:
        return "Гипонимов не найдено."

    # Улучшенное переранжирование с использованием target_word из трекера
    if reranking and results and tracker and tracker.target_word:
        try:
            candidates = tuple(item['name'] for item in results)

            # Адаптивный трешхолд (зависит от глубины понятия)
            gen = wordnet.find_generation(node_id)
            if gen >= 0:
                threshold = min(log2(gen + 1), 5)
            
            rerank_request = RerankerRequest(
                target=tracker.target_word,
                candidates=candidates,
                threshold=threshold/5.0,
                rel='hyponym',
                sense=sense
            )
            
            reranked = await reranker(rerank_request)
            
            reranked_names = {item['candidate'] for item in reranked}
            results = [item for item in results if item['name'] in reranked_names]
            
            logger.info(f"Реранкинг для '{tracker.target_word}' сократил результаты с {len(candidates)} до {len(results)}")
            
        except Exception as e:
            logger.error(f"Ошибка реранкинга: {e}")

    # Отслеживание выбранного синсета
    if tracker and node_id:
        tracker.add_synset(node_id, "get_hyponyms", {"node_id": node_id, "reranking": reranking, "interpreting": sense, "threshold": threshold})
    
    markdown = f"**Найдено гипонимов: {len(results)}**\n"
    
    for i, item in enumerate(results, 1):
        markdown += f"### {i}. {item['name']} `{item['id']}`\n"
        
        if item.get('definition'):
            markdown += f"**Определение:** {item['definition']}\n"
        
        words = item['words'][:5]
        words_str = "; ".join(words)
        if len(item['words']) > 5:
            words_str += f" *(+{len(item['words']) - 5} ещё)*"
        markdown += f"**Слова:** {words_str}\n"
        
        if item['hyponyms']:
            hyponyms_preview = "; ".join(item['hyponyms'][:10])
            if len(item['hyponyms']) > 10:
                hyponyms_preview += f" *(+{len(item['hyponyms']) - 10} ещё)*"
            markdown += f"**Гипонимы ({len(item['hyponyms'])}):** {hyponyms_preview}\n"
        else:
            markdown += f"**Гипонимов:** нет (конечный узел)\n"
        
        markdown += "---\n\n"
    
    return markdown


async def get_hypernyms(node_id, reranking=False, sense: Optional[str] = None, context='', threshold=0, tracker: SynsetTracker = None):
    """Tool function for getting hypernyms and formatting as markdown"""
    if node_id == 'null':
        node_id = None
    
    results = wordnet.get_hypernyms(node_id, pos='N')

    if not results:
        return "Гиперонимов не найдено."
    
    if reranking and results and tracker and tracker.target_word:
        try:
            candidates = tuple(item['name'] for item in results)

            # Адаптивный трешхолд (зависит от глубины понятия)
            gen = wordnet.find_generation(node_id)
            if gen >= 0:
                threshold = min(log2(gen + 1), 5)
            
            rerank_request = RerankerRequest(
                target=tracker.target_word,
                candidates=candidates,
                threshold=threshold/5.0,
                rel='hyponym',
                sense=sense
            )
            
            reranked = await reranker(rerank_request)
            
            reranked_names = {item['candidate'] for item in reranked}
            results = [item for item in results if item['name'] in reranked_names]
            
            logger.info(f"Реранкинг для '{tracker.target_word}' сократил результаты с {len(candidates)} до {len(results)}")
            
        except Exception as e:
            logger.error(f"Ошибка реранкинга: {e}")

    # Отслеживание выбранного синсета
    if tracker and node_id:
        tracker.add_synset(node_id, "get_hypernyms", {"node_id": node_id})
    
    markdown = f"**Найдено гиперонимов: {len(results)}**\n"
    
    for i, item in enumerate(results, 1):
        markdown += f"### {i}. {item['name']} `{item['id']}`\n"
        
        if item.get('definition'):
            markdown += f"**Определение:** {item['definition']}\n"
        
        words = item['words'][:5]
        words_str = "; ".join(words)
        if len(item['words']) > 5:
            words_str += f" *(+{len(item['words']) - 5} ещё)*"
        markdown += f"**Слова:** {words_str}\n"
        
        if item['hypernyms']:
            hypernyms_preview = "; ".join(item['hypernyms'][:10])
            if len(item['hypernyms']) > 10:
                hypernyms_preview += f" *(+{len(item['hypernyms']) - 10} ещё)*"
            markdown += f"**Гиперонимы ({len(item['hypernyms'])}):** {hypernyms_preview}\n"
        else:
            markdown += f"**Гиперонимов:** нет (корневой узел)\n"
        
        markdown += "---\n\n"
    
    return markdown


def save_tracking_data(tracking_data: dict, output_file: str):
    """Сохранить данные отслеживания в файл"""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Если файл уже существует, загружаем и добавляем новые данные
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if isinstance(existing_data, list):
                existing_data.append(tracking_data)
            else:
                existing_data = [existing_data, tracking_data]
        else:
            existing_data = [tracking_data]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные отслеживания сохранены в {output_file}")
        
    except Exception as e:
        logger.error(f"Ошибка сохранения данных отслеживания: {e}")


def get_system_prompt(functions: list):
    """Выбрать подходящий системный промпт в зависимости от функций"""
    if set(functions) == {"get_hyponyms"}:
        return read_prompt('system_hyponym', logger)
    elif set(functions) == {"get_hypernyms"}:
        return read_prompt('system_hypernym', logger)
    elif set(functions) == {"get_hyponyms", "get_hypernyms"}:
        return read_prompt('system', logger)
    else:
        return read_prompt('system', logger)

    
available_tools = {
    "get_hyponyms": get_hyponyms,
    "get_hypernyms": get_hypernyms
}

    
async def process_prediction(text: str, max_iterations: int, temperature: float, top_p: float,
                                   reranking: bool, interpreting: bool, functions: list, output_file: str = None, 
                                   start_node_id: str = None):
    """Process prediction without streaming and optional start node"""
    
    tracker = SynsetTracker()
    tracker.set_target(extract_target_word(text))
    
    # Подготавливаем пользовательское сообщение с учетом стартового узла
    sense = await interpreter(InterpreterRequest(context=text)) if interpreting else None
    user_message = await prepare_start_user_message(text, start_node_id, reranking=reranking, sense=sense, tracker=tracker)
    
    system_prompt = get_system_prompt(functions)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    # Если есть стартовый узел, добавляем его в трекер
    if start_node_id:
        tracker.add_synset(start_node_id, "start_node", {"node_id": start_node_id})

    # Получить правильный набор инструментов из tools.py
    if set(functions) == {"get_hyponyms"}:
        filtered_tools = available_tools_config['hyponym_only']
    elif set(functions) == {"get_hypernyms"}:
        filtered_tools = available_tools_config['hypernym_only'] 
    else:
        filtered_tools = available_tools_config['hyponym_only'] + available_tools_config['hypernym_only']
    
    final_result = None
    final_synset = None
    iteration_count = 0
    
    for i in range(max_iterations):
        iteration_count = i + 1
        
        try:
            response_obj = await oclient.chat.completions.create(
                model=os.environ['MODEL_NAME'],
                messages=messages,
                tools=filtered_tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=4000,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
        
        response_message = response_obj.choices[0].message
        messages.append(response_message.model_dump())
        
        # Check if this is the final response
        if not response_message.tool_calls:
            final_result = response_message.content.strip()
            final_synset = get_synset_id_from_response(final_result)
            logger.info(f'Извлечён id синсета из финального результата: {final_synset}')
            break
        
        # Process tool calls
        tool_messages = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            if not function_to_call or function_name not in functions:
                continue
            
            function_args = json.loads(tool_call.function.arguments)
            
            # Передаем трекер в функцию
            if function_name in ["get_hyponyms", "get_hypernyms"]:
                function_args.update({
                    'tracker': tracker,
                    'reranking': reranking,
                    'sense': sense,
                    'context': text
                })
            
            function_response = await function_to_call(**function_args)
            
            tool_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": function_response
            })
        
        messages.extend(tool_messages)
    
    if final_result is None:
        final_result = "Достигнут лимит итераций"
    
    # Получаем данные отслеживания
    if final_synset:
        tracker.add_synset(final_synset, "final_result", {"node_id": final_synset, "interpreting": sense, "reranking": reranking})
    tracking_data = tracker.get_tracking_data()
    tracking_data['final_result'] = final_result
    tracking_data['iterations'] = iteration_count
    tracking_data['timestamp'] = datetime.now().isoformat()
    
    # Сохраняем данные отслеживания если указан файл
    if output_file:
        save_tracking_data(tracking_data, output_file)
    
    return {
        "result": final_result,
        "iterations": iteration_count,
        "full_conversation": messages,
        "tracking_data": tracking_data
    }


async def process_prediction_stream(text: str, max_iterations: int, temperature: float, top_p: float,
                                   reranking: bool, interpreting: bool, functions: list, output_file: str = None, 
                                   start_node_id: str = None) -> AsyncGenerator[str, None]:
    """Process prediction with streaming and optional start node"""
    
    tracker = SynsetTracker()
    tracker.set_target(extract_target_word(text))
    
    # Подготавливаем пользовательское сообщение с учетом стартового узла
    sense = await interpreter(InterpreterRequest(context=text)) if interpreting else None
    user_message = await prepare_start_user_message(text, start_node_id, reranking=reranking, sense=sense, tracker=tracker)
    
    system_prompt = get_system_prompt(functions)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    # Фильтруем доступные инструменты согласно параметрам
    if set(functions) == {"get_hyponyms"}:
        filtered_tools = available_tools_config['hyponym_only']
    elif set(functions) == {"get_hypernyms"}:
        filtered_tools = available_tools_config['hypernym_only'] 
    else:
        filtered_tools = available_tools_config['hyponym_only'] + available_tools_config['hypernym_only']
    
    for i in range(max_iterations):
        yield f"data: {json.dumps({'type': 'iteration', 'iteration': i + 1}, ensure_ascii=False)}\n\n"
        
        try:
            response_obj = await oclient.chat.completions.create(
                model=os.environ['MODEL_NAME'],
                messages=messages,
                tools=filtered_tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=4000,
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            return
        
        response_message = response_obj.choices[0].message
        
        # Send assistant's thought if present
        if response_message.content:
            yield f"data: {json.dumps({'type': 'thought', 'content': response_message.content.strip()}, ensure_ascii=False)}\n\n"
        
        messages.append(response_message.model_dump())
        
        # Check if this is the final response
        if not response_message.tool_calls:
            final_result = response_message.content.strip()
            final_synset = get_synset_id_from_response(final_result)
            logger.info(f'Извлечён id синсета из финального результата: {final_synset}')
            
            # Получаем данные отслеживания
            if final_synset:
                tracker.add_synset(final_synset, "final_result", {"node_id": final_synset, "reranking": reranking})
            tracking_data = tracker.get_tracking_data()
            tracking_data['final_result'] = final_result
            tracking_data['iterations'] = i + 1
            tracking_data['timestamp'] = datetime.now().isoformat()
            
            # Сохраняем данные отслеживания если указан файл
            if output_file:
                save_tracking_data(tracking_data, output_file)
                yield f"data: {json.dumps({'type': 'tracking_saved', 'file': output_file, 'selections_count': len(tracking_data['selected_synsets'])}, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'type': 'final', 'result': final_result, 'tracking_data': tracking_data}, ensure_ascii=False)}\n\n"
            return
        
        # Process tool calls
        tool_messages = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            if not function_to_call or function_name not in functions:
                continue
            
            function_args = json.loads(tool_call.function.arguments)
            
            # Get node name for display
            node_name = 'root'
            if function_args.get('node_id') is not None and function_args['node_id'].lower() != 'none':
                if function_args['node_id'] in wordnet.synsets:
                    node_name = wordnet.synsets[function_args['node_id']].synset_name
            
            yield f"data: {json.dumps({'type': 'tool_call', 'function': function_name, 'args': function_args, 'node_name': node_name}, ensure_ascii=False)}\n\n"
            
            # Передаем трекер в функцию
            if function_name in ["get_hyponyms", "get_hypernyms"]:
                function_args.update({
                    'tracker': tracker,
                    'reranking': reranking,
                    'sense': sense,
                    'context': text
                })
            
            function_response = await function_to_call(**function_args)
            
            # Send the function response (markdown) to client
            yield f"data: {json.dumps({'type': 'tool_response', 'content': function_response}, ensure_ascii=False)}\n\n"
            
            tool_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": function_response
            })
        
        messages.extend(tool_messages)
        await asyncio.sleep(0.01)
    
    yield f"data: {json.dumps({'type': 'error', 'message': 'Достигнут лимит итераций'}, ensure_ascii=False)}\n\n"


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Regular prediction endpoint that returns the final result."""
    if '<predict_kb>' not in request.text or '</predict_kb>' not in request.text:
        raise HTTPException(status_code=400, detail="Text must contain <predict_kb>...</predict_kb> tags")
    
    result = await process_prediction(
        text=request.text,
        max_iterations=request.max_iterations,
        temperature=request.temperature,
        top_p=request.top_p,
        reranking=request.reranking,
        interpreting=request.interpreting,
        functions=request.functions,
        start_node_id=request.start_node_id,
        output_file=request.output_file
    )
    
    return result


@app.post("/predict/stream")
async def predict_stream(request: PredictionRequest):
    """Streaming prediction endpoint that returns the process in real-time."""
    if '<predict_kb>' not in request.text or '</predict_kb>' not in request.text:
        msg = "Text must contain <predict_kb>...</predict_kb> tags"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)
    
    return StreamingResponse(
        process_prediction_stream(
            text=request.text,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            top_p=request.top_p,
            reranking=request.reranking,
            interpreting=request.interpreting,
            functions=request.functions,
            start_node_id=request.start_node_id,
            output_file=request.output_file
        ),
        media_type="text/event-stream"
    )




@app.get("/health")
async def health():
    """Health check endpoint"""
    result = {"status": "ok", "wordnet_loaded": len(wordnet.synsets) > 0}
    logger.info("Health check %s", pformat(result))
    return result