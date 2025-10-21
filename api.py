from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from taxoenrich.core import RuWordNet
import json
from openai import OpenAI
import os
import asyncio
from typing import AsyncGenerator

app = FastAPI(title="RuWordNet Taxonomy Prediction API")

# Initialize RuWordNet
wordnet = RuWordNet('./wordnets/RuWordNet')

# Initialize OpenAI client
oclient = OpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    max_iterations: int = 50
    temperature: float = 0.5
    top_p: float = 0.95

class PredictionResponse(BaseModel):
    result: str
    iterations: int
    full_conversation: list = None


def get_hyponyms(node_id):
    """Tool function for getting hyponyms and formatting as markdown"""
    if node_id == 'null':
        node_id = None
    
    results = wordnet.get_hyponyms(node_id, pos='N')
    
    # Format as clean markdown
    if not results:
        return "Гипонимов не найдено."
    
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


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hyponyms",
            "description": "Navigate the RuWordNet taxonomy by retrieving hyponyms (more specific concepts) of a given synset. Returns formatted markdown with: synset name, ID, associated words, and list of child hyponyms. When node_id is null, returns all root nodes (top-level concepts). Use this to explore the taxonomy tree level by level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": ["string", "null"],
                        "description": "The synset ID to get hyponyms for. Use 'null' or null to retrieve all root nodes (top-level concepts in the taxonomy). Use specific synset ID like '123456-N' to get its children.",
                    },
                },
                "required": ["node_id"],
            },
        },
    }
]

system_prompt = '''Ты - экспертная система для классификации понятий в таксономии RuWordNet. Твоя задача - точно определить место понятия в иерархии, используя контекст и семантический анализ.

═══════════════════════════════════════════════════════════════════

📋 ФОРМАТ ВХОДНЫХ ДАННЫХ:
Ты получишь текст с понятием в тегах <predict_kb>...</predict_kb>. Анализируй:
1. Само понятие (слово/словосочетание)
2. Контекст использования в тексте
3. Семантическое значение в данном контексте

═══════════════════════════════════════════════════════════════════

🎯 ПРОЦЕСС ПРИНЯТИЯ РЕШЕНИЯ:

ШАГ 1: АНАЛИЗ ПОНЯТИЯ
- Определи точное значение понятия из контекста
- Выяви ключевые семантические характеристики
- Сформулируй для себя, к какой категории оно относится

ШАГ 2: НАВИГАЦИЯ ПО ТАКСОНОМИИ
- ОБЯЗАТЕЛЬНО начни с get_hyponyms(node_id=null) - просмотри корневые узлы
- Выбери наиболее релевантную ветку на основе семантики
- Двигайся от общего к частному, углубляясь в иерархию
- На каждом уровне анализируй:
  • Соответствуют ли слова узла понятию?
  • Является ли узел гиперонимом (более общим понятием)?
  • Есть ли среди гипонимов узла более подходящие варианты?

ШАГ 3: КРИТЕРИИ ВЫБОРА
Используй ТЕСТ ГИПЕРОНИМА: "Является ли найденный узел гиперонимом понятия?"
- ДА → Исследуй гипонимы этого узла глубже
- НЕТ → Вернись назад или исследуй другую ветку

ШАГ 4: ПРОВЕРКА АЛЬТЕРНАТИВ
- Исследуй минимум 2-3 релевантные ветки
- Сравни найденные варианты
- Выбери наиболее точное соответствие

═══════════════════════════════════════════════════════════════════

✅ ОКОНЧАТЕЛЬНОЕ РЕШЕНИЕ (один из трех вариантов):

1️⃣ "include in {synset_id} ({synset_name})"
   КОГДА ИСПОЛЬЗОВАТЬ:
   - Понятие - это СИНОНИМ существующего слова в синсете
   - Понятие обозначает ТО ЖЕ САМОЕ, что и узел
   - Понятие - альтернативное название того же объекта/явления
   
   ПРИМЕРЫ:
   - "автомобиль" → include in synset_машина
   - "компьютер" → include in synset_ЭВМ
   
   ПРОВЕРКА: Можно ли заменить понятие на слова из синсета без потери смысла?

2️⃣ "hyponym of {synset_id} ({synset_name})"
   КОГДА ИСПОЛЬЗОВАТЬ:
   - Понятие - это БОЛЕЕ КОНКРЕТНЫЙ вид узла
   - Понятие является ЧАСТНЫМ СЛУЧАЕМ узла
   - Верно утверждение: "{понятие} - это (один из видов) {узел}"
   
   ПРИМЕРЫ:
   - "грузовик" → hyponym of synset_автомобиль
   - "ноутбук" → hyponym of synset_компьютер
   
   ПРОВЕРКА: Является ли узел корректным гиперонимом (родителем) для понятия?

3️⃣ "not_found"
   КОГДА ИСПОЛЬЗОВАТЬ:
   - Понятие не вписывается ни в одну существующую категорию
   - Понятие слишком специфично и требует нового верхнеуровневого узла
   - После тщательного поиска не найдено подходящего места
   
   ⚠️ ВНИМАНИЕ: Используй only если исследовал все релевантные ветки!

═══════════════════════════════════════════════════════════════════

📌 ВАЖНЫЕ ПРАВИЛА:

✓ КОНТЕКСТ: Всегда учитывай, как понятие используется в тексте
✓ ТОЧНОСТЬ: Ищи максимально конкретный узел, не останавливайся на общих
✓ ГЛУБИНА: Углубляйся в иерархию - более глубокие узлы обычно точнее. Не бойся выбирать общие узлы, если однозначно можно сказать, что оцениваемое понятие является этим общим понятием.
✓ АЛЬТЕРНАТИВЫ: Проверяй несколько веток перед окончательным решением
✓ СЕМАНТИКА: Фокусируйся на значении, а не только на словах

✗ НЕ ПОВТОРЯЙ вызовы get_hyponyms для одного узла
✗ НЕ СПЕШИ с решением - исследуй достаточно глубоко
✗ НЕ ИГНОРИРУЙ контекст - он критичен для понимания значения
✗ НЕ ВЫБИРАЙ слишком общие узлы - ищи конкретные

═══════════════════════════════════════════════════════════════════

💭 РАЗМЫШЛЕНИЯ:
В **начале работы** проанализируй исследуемое понятие **в 1-2 абзаца**.
Затем, в каждой итерации кратко описывай:
- В случае, если ты выбираешь один из текущих гипонимов - очень краткое обоснование **максимум в одно предложение**, почему выбрал именно его.
- В случае, если ты решил изменить ветку для исследования (взяв не текущий гипоним) - требуется небольшое обоснование **максимум в 1 абзац**.
- Старайся быть лаконичным, кратким, по делу.

Если гипотеза не подтверждается - смело исследуй другие ветки!

═══════════════════════════════════════════════════════════════════

🎓 ФИНАЛЬНАЯ ПРОВЕРКА перед ответом:
1. Прочитал ли я контекст и понял значение?
2. Исследовал ли я достаточно веток таксономии?
3. Является ли выбранный узел корректным гиперонимом/синонимом?
4. Нет ли более конкретного подходящего узла глубже?
5. Уверен ли я в своем решении?

Только после ответа "да" на все вопросы - давай окончательный ответ!
'''

available_tools = {
    "get_hyponyms": get_hyponyms
}


def process_prediction(text: str, max_iterations: int, temperature: float, top_p: float):
    """Process prediction without streaming"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    final_result = None
    iteration_count = 0
    
    for i in range(max_iterations):
        iteration_count = i + 1
        
        try:
            response_obj = oclient.chat.completions.create(
                model='Qwen3-235B-A22B-Instruct-2507',
                messages=messages,
                tools=tools,
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
            break
        
        # Process tool calls
        tool_messages = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            if not function_to_call:
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


async def process_prediction_stream(text: str, max_iterations: int, temperature: float, top_p: float) -> AsyncGenerator[str, None]:
    """Process prediction with streaming"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    for i in range(max_iterations):
        yield f"data: {json.dumps({'type': 'iteration', 'iteration': i + 1}, ensure_ascii=False)}\n\n"
        
        try:
            response_obj = oclient.chat.completions.create(
                model='Qwen3-235B-A22B-Instruct-2507',
                messages=messages,
                tools=tools,
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
            yield f"data: {json.dumps({'type': 'final', 'result': response_message.content.strip()}, ensure_ascii=False)}\n\n"
            return
        
        # Process tool calls
        tool_messages = []
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            
            if not function_to_call:
                continue
            
            function_args = json.loads(tool_call.function.arguments)
            
            # Get node name for display
            node_name = 'root'
            if function_args['node_id'] is not None and function_args['node_id'].lower() != 'none':
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
        top_p=request.top_p
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
    """
    if '<predict_kb>' not in request.text or '</predict_kb>' not in request.text:
        raise HTTPException(status_code=400, detail="Text must contain <predict_kb>...</predict_kb> tags")
    
    return StreamingResponse(
        process_prediction_stream(
            text=request.text,
            max_iterations=request.max_iterations,
            temperature=request.temperature,
            top_p=request.top_p
        ),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "wordnet_loaded": len(wordnet.synsets) > 0}