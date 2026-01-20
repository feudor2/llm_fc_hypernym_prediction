from taxoenrich.core import RuWordNet

wordnet = RuWordNet('wordnets/RuWordNet')

import json
from openai import OpenAI
import os

oclient = OpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

def get_hyponyms(node_id):
    if node_id == 'null':
        node_id = None
    return wordnet.get_hyponyms(node_id, pos='N')

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hyponyms",
            "description": "Navigate the RuWordNet taxonomy by retrieving hyponyms (more specific concepts) of a given synset. When node_id is None, returns all root nodes (top-level concepts without parents). Each returned hyponym includes its name, associated words, unique ID, hyponyms (names only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": ["string", "null"],
                        "description": "The synset ID to get hyponyms for. Use null to retrieve all root nodes (top-level concepts in the taxonomy). Example: '123456-N' for a specific synset.",
                    },
                },
                "required": ["node_id"],
            },
        },
    }
]

system_prompt = '''Ты - интеллектуальный ассистент базы знаний для таксономии RuWordNet. Твоя задача - проанализировать текст и определить, куда следует поместить отмеченное понятие в таксономической иерархии.

ФОРМАТ ВХОДНЫХ ДАННЫХ:
Ты получишь текст с понятием, отмеченным тегами <predict_kb>...</predict_kb>. Это отмеченное понятие представляет собой слово или словосочетание, которое необходимо классифицировать в таксономии.

ТВОИ ВОЗМОЖНОСТИ:
1. Ты сначала думаешь над текущей информацией, запросе и известных тебе узлах в графе. Думаешь ты в каждом своем шаге. Если в текущем шаге все понятно и глубокие размышления не нужны, опиши в одном предложении текущий выбор.
2. Ты можешь вызывать инструмент `get_hyponyms` для навигации по таксономии и изучения синсетов и их гипонимов.
3. Ты должен в итоге принять окончательное решение о размещении отмеченного понятия

СТРАТЕГИЯ НАВИГАЦИИ:
- Первым вызовом get_hyponyms должен быть вызов с node_id=null, чтобы увидеть корневые понятия
- Перемещайся по таксономии, вызывая get_hyponyms с конкретными ID синсетов
- Систематически исследуй таксономию, чтобы найти наиболее подходящее место для отмеченного понятия

ОКОНЧАТЕЛЬНОЕ РЕШЕНИЕ:
После того как ты достаточно изучил таксономию, ты ДОЛЖЕН завершить работу ОДНИМ из этих трех ответов:

1. "not_found" - Когда отмеченное понятие не подходит ни к одному месту в таксономии или представляет собой совершенно новое понятие верхнего уровня

2. "include in {synset_id}" - Когда отмеченное понятие является синонимом или альтернативным термином для существующего синсета (или существующего термина). Понятие должно быть добавлено как новое слово в этот синсет.
   Пример: "include in 12345-N (*synset name*)"

3. "hyponym of {synset_id}" - Когда отмеченное понятие представляет более специфический тип (гипоним) существующего понятия и должно быть добавлено как новый синсет под ним.
   Пример: "hyponym of 12345-N (*synset name*)"

ВАЖНЫЕ ПРАВИЛА:
- Методично перемещайся по таксономии, используя get_hyponyms. Тебе запрещено вызывать get_hyponyms с узлами, с которыми ранее уже был осуществел вызов.
- Тщательно рассматривайте семантические связи перед принятием решения
- Выбирайте наиболее конкретного подходящего родителя при создании гипонима
- Если понятие явно принадлежит существующему синсету как синоним, используйте "include in"
- Если понятие является более конкретным типом чего-либо, используйте "hyponym of"
- Используйте "not_found" только когда понятие действительно не вписывается в существующую таксономию
- Ты можешь вызывать get_hyponyms для предыдущих узлов, если вам нужно выбрать другой путь
- Ты тщательно выбираешь нужный узел в графе, поэтому лучше заглядывай в дополнительные узлы, чтобы убедиться, что итоговый выбор корректный.
- Перед итоговым ответом подумай, можно ли сказать, что entity is a hypernym (где entity это выделенное в тексте понятие, а hypernym найденный узел)
- В процессе размышлений ты можешь продолжить исследовать граф, если не нашел нужных узлов.
'''

query = '''Каждое лето группы энтузиастов испытывают себя и отправляются на поиски снега и льда. Чаще всего их называют альпинисты, и они в любое время года не против пересечь ледник или тропить по снегу до вершины. Храбрые профи даже готовы лезть по скалам со льдом, выбирая запредельно сложные маршруты. Горные туристы тоже с удовольствием гуляют среди вечной мерзлоты на высотах более 4000 метров над уровнем моря. И всем им требуется надёжное сцепление на скользкой поверхности льда.

Итальянский кузнец и <predict_kb>основатель</predict_kb> легендарной альпинистской компании Генри Гривель более 100 лет назад снабдил одних из первых восходителей прообразом того, что сейчас называют кошками. Устройства были больше похожи на ряд соединённых скоб с заострёнными шипами и ремнями для крепления. Они изменили тактику передвижения по снежно-ледовому склону и значительно расширили возможности спортсменов.

С тех времён модели заметно усовершенствовали, но по-прежнему это изделия из металла, которые крепятся к ботинкам, вгрызаются в лёд и держат на снежном рельефе'''

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
]

available_tools = {
    "get_hyponyms": get_hyponyms
}

max_iterations = 50 # Защита от бесконечного цикла
for i in range(max_iterations):
    print(f"\n===== Итерация {i+1} =====\n")
    try:
        response_obj = oclient.chat.completions.create(
            model='Qwen3-235B-A22B-Instruct-2507',
            messages=messages,
            tools=tools,
            temperature=0.5,
            top_p=0.95,
            max_tokens=4000,
        )
    except Exception as e:
        print(e)
        break
    
    # В vLLM ответ приходит как объект, а не словарь, поэтому доступ через точку
    # Если у вас словарь, используйте response_obj['choices'][0]['message']
    response_message = response_obj.choices[0].message
    if response_message.content is not None:
        print(response_message.content.strip())
    # Добавляем ответ ассистента в историю, чтобы он помнил, что делал
    messages.append(response_message.model_dump())

    # Проверяем, есть ли запрос на вызов функции
    if not response_message.tool_calls:
        # Если tool_calls нет, значит это финальный текстовый ответ
        print(f"\n✅ Финальный ответ модели:\n{response_message.content}")
        break
    
    # Если есть tool_calls, выполняем их
    # Создаем временный список для сообщений с результатами
    tool_messages = []
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_tools.get(function_name)
        
        if not function_to_call:
             print(f"❌ Ошибка: Модель вызвала несуществующую функцию: {function_name}")
             # Можно добавить сообщение об ошибке и для модели
             continue

        function_args = json.loads(tool_call.function.arguments)
        node_name = 'root'
        if function_args['node_id'] is not None and function_args['node_id'].lower() != 'none':
            node_name = wordnet.synsets[function_args['node_id']].synset_name
        print(f"Вызов функции:\n{function_name}({function_args}/{node_name})")
        
        # Вызываем функцию и получаем результат
        function_response = function_to_call(**function_args)
        #print(json.dumps(function_response, ensure_ascii=False, indent=4))
        # Формируем сообщение с результатом для следующего запроса
        tool_messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": json.dumps(function_response, ensure_ascii=False) # Сериализуем результат в JSON строку
        })

    # Добавляем все сообщения с результатами инструментов в общую историю
    messages.extend(tool_messages)
else:
    print("\n⚠️ Достигнут лимит итераций. Цикл прерван.")