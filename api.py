import json
import os
import re
import asyncio
import logging

from datetime import datetime
from pprint import pformat
from typing import AsyncGenerator
from dotenv import load_dotenv

from pydantic import BaseModel
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from taxoenrich.core import RuWordNet
from utils import *
from io_utils import read_yaml
from tools import tools as available_tools_config
from reranker import Reranker, RerankerRequest

load_dotenv()
config = read_yaml('config.yml')

app = FastAPI(title="RuWordNet Taxonomy Prediction API")

# Initialize RuWordNet
wordnet = RuWordNet('./wordnets/RuWordNet')

# Initialize OpenAI client
oclient = OpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Initialize logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_dir = './logs'

logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
reranker = Reranker(oclient, os.environ['MODEL_NAME'], logger)


# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    max_iterations: int = 50
    temperature: float = 0.5
    top_p: float = 0.95
    # Новые параметры пайплайна
    reranking: bool = False
    functions: list = ["get_hyponyms"]

class PredictionResponse(BaseModel):
    result: str
    iterations: int
    full_conversation: list = None


def get_hyponyms(node_id, reranking=False, threshold=3):
    """Tool function for getting hyponyms and formatting as markdown"""
    if node_id == 'null':
        node_id = None
    
    results = wordnet.get_hyponyms(node_id, pos='N')

    # Format as clean markdown
    if not results:
        return "Гипонимов не найдено."

    # Если включено переранжирование и есть результаты, применяем его
    if reranking and results and len(results) > threshold:
        try:
            # Извлекаем имена синсетов как кандидатов
            candidates = tuple(item['name'] for item in results)
            
            # Получаем target word из контекста (это нужно передать в функцию)
            # Пока используем заглушку - в реальности target нужно получать из вызова
            target_word = "unknown"  # TODO: передавать target из вызывающего кода
            
            # Создаем запрос на переранжирование
            rerank_request = RerankerRequest(
                target=target_word,
                candidates=candidates,
                threshold=threshold/5.0,  # Конвертируем в шкалу 0-1
                rel='hyponym'
            )
            
            # Применяем переранжирование (синхронно)
            import asyncio
            reranked = asyncio.run(reranker(rerank_request))
            
            # Фильтруем результаты согласно переранжированию
            reranked_names = {item['candidate'] for item in reranked}
            results = [item for item in results if item['name'] in reranked_names]
            
            logger.info(f"Реранкинг сократил результаты с {len(candidates)} до {len(results)}")
            
        except Exception as e:
            logger.error(f"Ошибка реранкинга: {e}")
            # Продолжаем с исходными результатами в случае ошибки
    
    markdown = f"**Найдено гипонимов: {len(results)}**\n"
    
    for i, item in enumerate(results, 1):
        # Header with name and ID
        markdown += f"### {i}. {item['name']} `{item['id']}`\n"
        
        # Definition (if available)
        if item.get('definition'):
            markdown += f"**Определение:** {item['definition']}\n"
        
        # Words (limit to first 5 for brevity)
        words = item['words'][:5]
        words_str = "; ".join(words)
        if len(item['words']) > 5:
            words_str += f" *(+{len(item['words']) - 5} ещё)*"
        markdown += f"**Слова:** {words_str}\n"
        
        # Hyponyms (show count and first few names)
        if item['hyponyms']:
            hyponyms_preview = "; ".join(item['hyponyms'][:10])
            if len(item['hyponyms']) > 10:
                hyponyms_preview += f" *(+{len(item['hyponyms']) - 10} ещё)*"
            markdown += f"**Гипонимы ({len(item['hyponyms'])}):** {hyponyms_preview}\n"
        else:
            markdown += f"**Гипонимов:** нет (конечный узел)\n"
        
        markdown += "---\n\n"
    
    return markdown


def get_hypernyms(node_id):
    """Tool function for getting hypernyms and formatting as markdown"""
    if node_id == 'null':
        node_id = None
    
    results = wordnet.get_hypernyms(node_id, pos='N')

    # Format as clean markdown
    if not results:
        return "Гиперонимов не найдено."
    
    markdown = f"**Найдено гиперонимов: {len(results)}**\n"
    
    for i, item in enumerate(results, 1):
        # Header with name and ID
        markdown += f"### {i}. {item['name']} `{item['id']}`\n"
        
        # Definition (if available)
        if item.get('definition'):
            markdown += f"**Определение:** {item['definition']}\n"
        
        # Words (limit to first 5 for brevity)
        words = item['words'][:5]
        words_str = "; ".join(words)
        if len(item['words']) > 5:
            words_str += f" *(+{len(item['words']) - 5} ещё)*"
        markdown += f"**Слова:** {words_str}\n"
        
        # Hypernyms (show count and first few names)
        if item['hypernyms']:
            hypernyms_preview = "; ".join(item['hypernyms'][:10])
            if len(item['hypernyms']) > 10:
                hypernyms_preview += f" *(+{len(item['hypernyms']) - 10} ещё)*"
            markdown += f"**Гиперонимы ({len(item['hypernyms'])}):** {hypernyms_preview}\n"
        else:
            markdown += f"**Гиперонимов:** нет (корневой узел)\n"
        
        markdown += "---\n\n"
    
    return markdown


def get_system_prompt(functions: list):
    """Выбрать подходящий системный промпт в зависимости от функций"""
    if set(functions) == {"get_hyponyms"}:
        return read_prompt('system_hyponym', logger)
    elif set(functions) == {"get_hypernyms"}:
        return read_prompt('system_hypernym', logger)
    elif set(functions) == {"get_hyponyms", "get_hypernyms"}:
        return read_prompt('system', logger)
    else:
        # Fallback на обычный системный промпт
        return read_prompt('system', logger)
    

available_tools = {
    "get_hyponyms": get_hyponyms,
    "get_hypernyms": get_hypernyms
}

    
def process_prediction(text: str, max_iterations: int, temperature: float, top_p: float, 
                      reranking: bool, functions: list):
    """Process prediction without streaming"""
    
    # Выбрать подходящий системный промпт
    system_prompt = get_system_prompt(functions)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    # Получить правильный набор инструментов из tools.py
    if set(functions) == {"get_hyponyms"}:
        filtered_tools = available_tools_config['hyponym_only']
    elif set(functions) == {"get_hypernyms"}:
        filtered_tools = available_tools_config['hypernym_only'] 
    else:
        # Объединить оба набора инструментов
        filtered_tools = available_tools_config['hyponym_only'] + available_tools_config['hypernym_only']
    
    final_result = None
    iteration_count = 0
    
    for i in range(max_iterations):
        iteration_count = i + 1
        
        try:
            response_obj = oclient.chat.completions.create(
                model=os.environ['MODEL_NAME'],
                messages=messages,
                tools=filtered_tools,  # Используем фильтрованный список
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
            
            # Применение переранжирования если включено
            if reranking and final_result:
                logger.info("Applying reranking to final result")
                # Здесь будет логика переранжирования когда API будет готово
                # final_result = apply_reranking(final_result)
            
            break
        
        # Process tool calls
        tool_messages = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            if not function_to_call or function_name not in functions:
                continue
            
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            
            # Return markdown directly, not as JSON
            tool_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": function_response  # Already formatted as markdown
            })
        
        messages.extend(tool_messages)
    
    if final_result is None:
        final_result = "Достигнут лимит итераций"
    
    return {
        "result": final_result,
        "iterations": iteration_count,
        "full_conversation": messages
    }


async def process_prediction_stream(text: str, max_iterations: int, temperature: float, top_p: float,
                                   reranking: bool, functions: list) -> AsyncGenerator[str, None]:
    """Process prediction with streaming"""
    
    # Логирование параметров пайплайна
    logger.info(f"Pipeline parameters - reranking: {reranking}, functions: {functions}")
    system_prompt = get_system_prompt(functions)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    # Фильтруем доступные инструменты согласно параметрам
    if set(functions) == {"get_hyponyms"}:
        filtered_tools = available_tools_config['hyponym_only']
    elif set(functions) == {"get_hypernyms"}:
        filtered_tools = available_tools_config['hypernym_only'] 
    else:
        # Объединить оба набора инструментов
        filtered_tools = available_tools_config['hyponym_only'] + available_tools_config['hypernym_only']
    
    for i in range(max_iterations):
        yield f"data: {json.dumps({'type': 'iteration', 'iteration': i + 1}, ensure_ascii=False)}\n\n"
        
        try:
            response_obj = oclient.chat.completions.create(
                model=os.environ['MODEL_NAME'],
                messages=messages,
                tools=filtered_tools,  # Используем фильтрованный список
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
            
            # Применение переранжирования если включено
            if reranking and final_result:
                yield f"data: {json.dumps({'type': 'reranking', 'message': 'Применение переранжирования...'}, ensure_ascii=False)}\n\n"
                # Здесь будет логика переранжирования когда API будет готово
                # final_result = apply_reranking(final_result)
            
            yield f"data: {json.dumps({'type': 'final', 'result': final_result}, ensure_ascii=False)}\n\n"
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
            
            function_response = function_to_call(**function_args)
            
            # Send the function response (markdown) to client
            yield f"data: {json.dumps({'type': 'tool_response', 'content': function_response}, ensure_ascii=False)}\n\n"
            
            # Return markdown directly, not as JSON
            tool_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "content": function_response  # Already formatted as markdown
            })
        
        messages.extend(tool_messages)
        await asyncio.sleep(0.01)  # Small delay for streaming effect
    
    yield f"data: {json.dumps({'type': 'error', 'message': 'Достигнут лимит итераций'}, ensure_ascii=False)}\n\n"


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Regular prediction endpoint that returns the final result.
    
    The text must contain exactly one occurrence of <predict_kb>...</predict_kb> tags.
    """
    if '<predict_kb>' not in request.text or '</predict_kb>' not in request.text:
        raise HTTPException(status_code=400, detail="Text must contain <predict_kb>...</predict_kb> tags")
    
    result = process_prediction(
        text=request.text,
        max_iterations=request.max_iterations,
        temperature=request.temperature,
        top_p=request.top_p,
        reranking=request.reranking,
        functions=request.functions
    )
    
    return result


@app.post("/predict/stream")
async def predict_stream(request: PredictionRequest):
    """
    Streaming prediction endpoint that returns the process in real-time.
    
    The text must contain exactly one occurrence of <predict_kb>...</predict_kb> tags.
    
    Stream format (Server-Sent Events):
    - type: 'iteration' - New iteration started
    - type: 'thought' - Assistant's reasoning
    - type: 'tool_call' - Function call made
    - type: 'final' - Final result
    - type: 'error' - Error occurred
    - type: 'reranking' - Reranking is being applied
    """
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
            functions=request.functions
        ),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    result = {"status": "ok", "wordnet_loaded": len(wordnet.synsets) > 0}
    logger.info("Health check %s", pformat(result))
    return result