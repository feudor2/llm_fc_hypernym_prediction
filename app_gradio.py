import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

import re
import gradio as gr
import requests
import json
from typing import Generator
import threading
from queue import Queue
import time
import os
import glob
from pathlib import Path
from pprint import pformat

# Импортируем функции для работы с данными
from data_processing import load_dataset, load_corpus_text, load_start_nodes, convert_paths
from io_utils import read_json

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

http_client = httpx.AsyncClient(timeout=300.0)

def process_text_stream(text: str, max_iterations: int, temperature: float, top_p: float, 
                       reranking: bool, interpreting: bool, functions: list, output_file: str = None, start_node_id: str = None):
    """Process text using the streaming API endpoint with optional start node"""
    
    # Validate input
    if '<predict_kb>' not in text or '</predict_kb>' not in text:
        yield "❌ Ошибка: Текст должен содержать теги <predict_kb>...</predict_kb>", ""
        return
    
    valid_functions = ["get_hyponyms", "get_hypernyms"]
    functions = [f for f in functions if f in valid_functions]
    
    if not functions:
        functions = ["get_hyponyms"]
    
    api_url = "http://localhost:8500"
    
    # Prepare request with optional start node
    endpoint = f"{api_url.rstrip('/')}/predict/stream"
    payload = {
        "text": text,
        "max_iterations": max_iterations,
        "temperature": temperature,
        "top_p": top_p,
        "reranking": reranking,
        "interpreting": interpreting,
        "functions": functions,
        "output_file": output_file
    }
    
    # Добавляем стартовый узел если указан
    if start_node_id:
        payload["start_node_id"] = start_node_id
    
    # Остальная логика остается той же...
    
    process_log = ""
    final_result = ""

    logger.info(f'Sending request with payload {pformat(payload)}')
    
    try:
        # Make streaming request
        with requests.post(endpoint, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    # Parse SSE format
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        try:
                            data = json.loads(data_str)
                            event_type = data.get('type')
                            
                            if event_type == 'iteration':
                                iteration_info = f"\n{'='*54}\n🔄 Итерация {data['iteration']}\n{'='*54}\n\n"
                                process_log += iteration_info
                                yield process_log, final_result
                            
                            elif event_type == 'thought':
                                thought = f"💭 Размышление модели:\n{data['content']}\n\n"
                                process_log += thought
                                yield process_log, final_result
                            
                            elif event_type == 'tool_call':
                                tool_info = f"🔧 Вызов функции: {data['function']}\n"
                                tool_info += f"Аргументы: {json.dumps(data['args'], ensure_ascii=False)}\n"
                                tool_info += f"Узел: {data['node_name']}\n\n"
                                process_log += tool_info
                                yield process_log, final_result
                            
                            elif event_type == 'tool_response':
                                # Display the markdown formatted function response
                                response_info = f"📋 Результат функции:\n{data['content']}\n"
                                process_log += response_info
                                yield process_log, final_result
                            
                            elif event_type == 'tracking_saved':
                                tracking_info = f"💾 Данные отслеживания сохранены в {data['file']} (выбрано узлов: {data['selections_count']})\n"
                                process_log += tracking_info
                                yield process_log, final_result
                            
                            elif event_type == 'final':
                                final_result = f"✅ Финальный результат:\n\n{data['result']}"
                                
                                # Добавляем информацию об отслеживании если есть
                                if 'tracking_data' in data:
                                    tracking_data = data['tracking_data']
                                    tracking_info = f"\n\n📊 Статистика отслеживания:\n"
                                    tracking_info += f"• Целевое слово: {tracking_data.get('target_word', 'Неизвестно')}\n"
                                    tracking_info += f"• Выбрано узлов: {tracking_data.get('total_selections', 0)}\n"
                                    
                                    if tracking_data.get('selected_synsets'):
                                        tracking_info += f"• Выбранные синсеты:\n"
                                        for synset in tracking_data['selected_synsets']:
                                            tracking_info += f"  - {synset['synset_id']} ({synset['function']})\n"
                                    
                                    final_result += tracking_info
                                
                                process_log += f"\n{final_result}\n"
                                process_log += f"\n{'='*54}\n✔️ Анализ завершен\n{'='*54}\n"
                                yield process_log, final_result
                                return
                            
                            elif event_type == 'error':
                                error_msg = f"❌ Ошибка: {data['message']}\n"
                                process_log += error_msg
                                yield process_log, final_result
                                return
                        
                        except json.JSONDecodeError:
                            continue
        
    except requests.exceptions.Timeout:
        yield "❌ Ошибка: Превышено время ожидания ответа от сервера", ""
    except requests.exceptions.ConnectionError:
        yield f"❌ Ошибка: Не удалось подключиться к серверу. Убедитесь, что API запущен на {api_url}", ""
    except requests.exceptions.HTTPError as e:
        yield f"❌ Ошибка HTTP: {e.response.status_code} - {e.response.text}", ""
    except Exception as e:
        yield f"❌ Ошибка: {str(e)}", ""


def safe_file_path(file_obj):
    """Безопасно получить путь к файлу"""
    if file_obj is None:
        return None
    if hasattr(file_obj, 'name'):
        return file_obj.name
    return str(file_obj)


def process_dataset_item(
        dataset_path: str, corpus_folder: str, word: str, 
        max_iterations: int, temperature: float, top_p: float, 
        reranking: bool, interpreting: bool, functions: list, 
        num_processes: int, start_nodes_folder: str, max_n_starting_nodes: int,
        parallel_mode: str, output_file: str = None
    ):
    """Process a specific word from dataset using corpus text"""
    try:
        # Load corpus text for the word
        dataset = convert_paths(load_dataset(dataset_path), 0)
        texts = [load_corpus_text(corpus_folder, item_path) for item_path in dataset[word]]
        if not texts:
            yield f"❌ Не найден текст для слова: {word}", ""
            return
        
        # Генерируем имя файла для отслеживания если не указан
        if not output_file:
            timestamp = int(time.time())
            output_file = f"tracking_results/single_word_{word}_{timestamp}.json"
        
        # Process using the streaming function
        for results in run_parallel_analysis(
            word, texts, max_iterations, temperature, top_p, reranking, interpreting, functions, output_file,
            num_processes, start_nodes_folder, max_n_starting_nodes, parallel_mode
        ):
            yield results
    except Exception as e:
        yield f"❌ Ошибка при обработке слова '{word}': {str(e)}", ""


def get_dataset_info(dataset_path: str):
    """Безопасно загружаем информацию о датасете"""
    if not dataset_path or not os.path.exists(dataset_path):
        return "Датасет не выбран или файл не найден", gr.update(choices=[], interactive=False), gr.update(maximum=1, value=1)
    
    try:
        dataset = load_dataset(dataset_path)
        words = list(dataset.keys())
        max_samples = len(words)
        info = f"📊 Загружен датасет: {len(words)} слов"
        
        return (
            info, 
            gr.update(choices=[], interactive=True, allow_custom_value=True),
            gr.update(maximum=max_samples, value=1, interactive=True),
            gr.update(maximum=max_samples, value=max_samples, interactive=True),
            max_samples
        )
    except Exception as e:
        return f"❌ Ошибка загрузки датасета: {str(e)}", gr.update(choices=[], interactive=False), gr.update(maximum=1, value=1), 1, 1

def search_words_in_dataset(dataset_path: str, search_query: str):
    """Search for words in dataset that match the query"""
    if not dataset_path or not search_query:
        return gr.update(value=search_query.upper(), choices=[])
    
    try:
        dataset = load_dataset(dataset_path)
        words = list(dataset.keys())
        
        # Поиск слов, содержащих запрос (регистронезависимо)
        search_query = search_query.upper().strip()
        matching_words = [word for word in words if search_query in word.upper()]
        
        # Ограничиваем результат до 50 слов для производительности
        matching_words = matching_words[:50]
        
        return gr.update(value=search_query.upper(), choices=matching_words)
    except Exception as e:
        logger.error(f"Ошибка поиска в датасете: {e}")
        return gr.update(value=search_query.upper(), choices=[])

def validate_word_in_dataset(dataset_path: str, word: str):
    """Validate that the word exists in dataset"""
    if not dataset_path or not word:
        return "Выберите слово из датасета"
    
    try:
        dataset = load_dataset(dataset_path)
        if word in dataset:
            return f"✅ Слово '{word}' найдено в датасете"
        else:
            # Предложить похожие слова
            words = list(dataset.keys())
            similar = [w for w in words if word.upper() in w.upper() or w.upper() in word.upper()][:5]
            if similar:
                return f"❌ Слово '{word}' не найдено. Возможно вы имели в виду: {', '.join(similar)}"
            else:
                return f"❌ Слово '{word}' не найдено в датасете"
    except Exception as e:
        return f"❌ Ошибка проверки: {str(e)}"

def load_word_text_from_corpus(dataset_path, corpus_folder: str, word: str, index: int = 0):
    """Load text for specific word from corpus"""
    if not dataset_path or not corpus_folder or not word:
        return "", gr.update(), 0, gr.update()
    
    dataset = load_dataset(dataset_path)
    paths = convert_paths({word: dataset[word]})[word]
    n_texts = len(paths)
    
    if paths:
        if index >= n_texts:
            index = 0
        elif index < 0:
            index = n_texts - 1
        try:
            text = load_corpus_text(corpus_folder, paths[index])
            logger.debug(f'Извлечённый текст {text}')
            interactive = len(paths) > 1
            return text, gr.update(interactive=interactive), str(index + 1), gr.update(interactive=interactive)
        except Exception as e:
            return f"❌ Ошибка чтения файла: {str(e)}", gr.update(), '0', gr.update()
    
    return f"❌ Для слова {word} не найдено текстов в корпусе", gr.update(), '0', gr.update()

async def process_dataset_batch_async(dataset_file, corpus_folder, sample_start, max_samples, num_processes, batch_size,
                                     max_iterations, temperature, top_p, reranking, interpreting, functions, 
                                     start_nodes_path, max_n_starting_nodes, parallel_mode, progress=None):
    """Асинхронная батч-обработка с поддержкой множественных процессов"""
    start_time = int(time.time())

    if not dataset_file or not corpus_folder:
        return "❌ Выберите датасет и папку корпуса"
    
    # Загружаем данные
    file_path = safe_file_path(dataset_file)
    dataset = convert_paths(load_dataset(file_path), 0)
    start_nodes_dict = load_start_nodes(start_nodes_path) if start_nodes_path and max_n_starting_nodes > 0 else {}
    
    words_to_process = list(dataset.keys())[sample_start-1:sample_start+max_samples-1]
    total_tasks = 0
    
    # Подсчитываем общее количество задач
    for word in words_to_process:
        if word in start_nodes_dict and max_n_starting_nodes > 0 and parallel_mode == "по стартовым узлам":
            word_nodes = start_nodes_dict[word][:max_n_starting_nodes]
            total_tasks += len(word_nodes)
        elif max_n_starting_nodes > 0 and parallel_mode == "по стартовым узлам":
            logger.warning(f'Word {word} not found in start nodes list ({list(start_nodes_dict.keys())[0]},...)')
        elif parallel_mode == "по текстам" and dataset[word]:
            total_tasks += len(dataset[word])
        elif parallel_mode == "по текстам":
            logger.warning(f'No texts for {word} found in the dataset')
        else:
            total_tasks += num_processes
    
    logger.info(f"Запуск {total_tasks} задач для {len(words_to_process)} слов")
    
    # Создаем семафор для ограничения одновременных запросов
    semaphore = asyncio.Semaphore(min(total_tasks, 10))  # Максимум 10 одновременных запросов
    
    async def process_single_task(word, text, start_node_id, task_id):
        async with semaphore:
            try:
                # Подготовка запроса
                payload = {
                    "text": text,
                    "max_iterations": max_iterations,
                    "temperature": temperature,
                    "top_p": top_p,
                    "reranking": reranking,
                    "interpreting": interpreting,
                    "functions": functions,
                    "output_file": f"tracking_results/batch_{start_time}/async_word_{word}_[{task_id}].json"
                }
                
                # Добавляем стартовый узел если есть
                if start_node_id:
                    payload["start_node_id"] = start_node_id
                
                # Асинхронный вызов API
                response = await http_client.post(
                    "http://localhost:8500/predict",
                    json=payload
                )
                response.raise_for_status()
                
                result_data = response.json()
                return {
                    "word": word,
                    "result": result_data.get("result"),
                    "iterations": result_data.get("iterations"),
                    "task_id": task_id,
                    "start_node_id": start_node_id
                }
                
            except Exception as e:
                logger.error(f"Ошибка обработки задачи {task_id} для слова {word}: {e}")
                return {
                    "word": word,
                    "error": str(e),
                    "task_id": task_id,
                    "start_node_id": start_node_id
                }
    
    # Обновляем прогресс
    if progress:
        progress(0.1, f"Создано {total_tasks} задач для {len(words_to_process)} слов")
    
    # Выполняем все задачи с обновлением прогресса
    completed_tasks = 0
    results = []

    tasks = []
    
    for w, word in enumerate(words_to_process):
        if word in start_nodes_dict and max_n_starting_nodes > 0:
            # Используем предзаданные узлы
            word_nodes = start_nodes_dict[word][:max_n_starting_nodes]
        else:
            word_nodes = []

        if parallel_mode == 'по текстам':
            for t, path in enumerate(dataset[word]):
                text = load_corpus_text(corpus_folder, path)
                if not text or "❌" in text:
                    continue
                if word_nodes:
                    for n, node_id in enumerate(word_nodes):
                        task_id = f'{w}_{t}_{n}_0'
                        tasks.append(process_single_task(word, text, node_id, task_id))
                else:
                    task_id = f'{w}_{t}__0'
                    tasks.append(process_single_task(word, text, None, task_id))
        else:
            text = load_corpus_text(corpus_folder, dataset[word][0])
            if not text or "❌" in text:
                continue
            if parallel_mode == 'по стартовым узлам':
                for n, node_id in enumerate(word_nodes):
                    task_id = f'{w}_0_{n}_0'
                    tasks.append(process_single_task(word, text, node_id, task_id))
            else:
                # Стандартный режим с несколькими процессами
                for i in range(num_processes):
                    task_id = f'{w}_0__{i}'
                    tasks.append(process_single_task(word, text, None, task_id))

    
    # Обрабатываем задачи батчами для обновления прогресса
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
        
        completed_tasks += len(batch_tasks)
        if progress:
            progress_value = 0.1 + (completed_tasks / total_tasks) * 0.8
            progress(progress_value, f"Выполнено {completed_tasks}/{total_tasks} задач")
    
    # Сохраняем результаты
    timestamp = int(time.time())
    output_file = f"test_results/async_batch_{len(words_to_process)}words_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Группируем результаты по словам
    results_by_word = {}
    for result in results:
        if isinstance(result, dict) and 'word' in result:
            word = result['word']
            if word not in results_by_word:
                results_by_word[word] = []
            results_by_word[word].append(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_by_word, f, ensure_ascii=False, indent=2)
    
    if progress:
        progress(1.0, "Обработка завершена")
    
    return f"✅ Асинхронная обработка завершена.\n📊 Выполнено {len(results)} задач для {len(words_to_process)} слов\n💾 Результаты: {output_file}"

def process_dataset_batch(dataset_file, corpus_folder, sample_start, max_samples, num_processes, batch_size,
                         max_iterations, temperature, top_p, reranking, interpreting, functions,
                         start_nodes_path, max_n_starting_nodes, parallel_mode, progress=gr.Progress()):
    """Wrapper для запуска асинхронной батч-обработки"""
    
    # Запускаем асинхронную функцию в новом event loop
    def run_async():
        try:
            # Создаем новый event loop для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                process_dataset_batch_async(
                    dataset_file, corpus_folder, sample_start, max_samples, num_processes, batch_size,
                    max_iterations, temperature, top_p, reranking, interpreting, functions,
                    start_nodes_path, max_n_starting_nodes, parallel_mode, progress
                )
            )
        finally:
            loop.close()
    
    # Выполняем в отдельном потоке чтобы не блокировать Gradio
    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_async)
        return future.result() 
    
def get_start_nodes_info(start_nodes_path: str):
    """Get information about start nodes file"""
    if not start_nodes_path or not os.path.exists(start_nodes_path):
        return "Файл со стартовыми узлами не выбран или не найден", gr.update(), gr.update()
    
    try:
        content = read_json(start_nodes_path)
        total_words = len(content)
        total_nodes = sum(len(nodes) for nodes in content.values())
        return f"📁 Загружен файл: {total_words} слов, {total_nodes} стартовых узлов", gr.update(interactive=True, maximum=3), gr.update(choices=['по стартовым узлам']+parallel_mode.choices, value='по стартовым узлам')
    except Exception as e:
        return f"❌ Ошибка чтения файла: {str(e)}", gr.update(), gr.update()


# Example text
example_text = '''Каждое лето группы энтузиастов испытывают себя и отправляются на поиски снега и льда. Чаще всего их называют альпинисты, и они в любое время года не против пересечь ледник или тропить по снегу до вершины. Храбрые профи даже готовы лезть по скалам со льдом, выбирая запредельно сложные маршруты. Горные туристы тоже с удовольствием гуляют среди вечной мерзлоты на высотах более 4000 метров над уровнем моря. И всем им требуется надёжное сцепление на скользкой поверхности льда.

Итальянский кузнец и основатель легендарной альпинистской компании Генри Гривель более 100 лет назад снабдил одних из первых восходителей прообразом того, что сейчас называют <predict_kb>кошками</predict_kb>. Устройства были больше похожи на ряд соединённых скоб с заострёнными шипами и ремнями для крепления. Они изменили тактику передвижения по снежно-ледовому склону и значительно расширили возможности спортсменов.

С тех времён модели заметно усовершенствовали, но по-прежнему это изделия из металла, которые крепятся к ботинкам, вгрызаются в лёд и держат на снежном рельефе'''

custom_css = """
.center-text {
    text-align: center;
    display: block; /* Ensures the text container behaves like a block element */
}
"""
# Create Gradio interface
with gr.Blocks(title="RuWordNet Taxonomy Prediction Client", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🔍 RuWordNet Taxonomy Prediction Client")
    gr.Markdown("""
    Этот интерфейс обращается к API сервису для анализа текста и определения места понятия в таксономии RuWordNet.
    
    **Инструкция:**
    1. Настройте параметры модели и пайплайна
    2. Выберите режим работы: "Ручной ввод" или "Датасет"
    3. Введите текст или выберите файлы
    4. Запустите анализ
    """)

    # Parameters section (shared)
    with gr.Accordion("⚙️ Параметры", open=False):
        with gr.Row():
            # Model parameters
            with gr.Column():
                gr.Markdown("**Параметры модели**")
                max_iterations = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Максимум итераций"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top P"
                )
            
            # Pipeline parameters
            with gr.Column():
                gr.Markdown("**Параметры пайплайна**")
                with gr.Row():
                    reranking = gr.Checkbox(
                        label="🔄 Переранжирование",
                        value=True
                    )
                    interpreting = gr.Checkbox(
                        label="🔍 Семантический анализ",
                        value=True,
                        interactive=True
                    )
                functions = gr.CheckboxGroup(
                    label="🔧 Функции",
                    choices=[
                        ("Получить гипонимы", "get_hyponyms"),
                        ("Получить гиперонимы", "get_hypernyms")
                    ],
                    value=["get_hyponyms"]
                )
                num_processes = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=3,
                    step=1,
                    label="🔧 Количество процессов",
                    info="Количество параллельных процессов для одного слова"
                )
                max_n_starting_nodes = gr.Slider(
                    minimum=0,
                    maximum=3,
                    value=0,
                    step=1,
                    label="📍 Количество стартовых узлов",
                    info="0 = стандартный режим, 1-3 = режим с предзаданными узлами",
                    interactive=False
                )
    
    # Mode selection
    with gr.Tab("🖊️ Ручной ввод"):
        # Text input
        text_input = gr.Textbox(
            label="📝 Входной текст",
            placeholder="Введите текст с тегами <predict_kb>...</predict_kb>",
            lines=10,
            value=example_text
        )
        
        # Output file for tracking
        manual_output_file = gr.Textbox(
            label="💾 Файл для сохранения отслеживания (опционально)",
            placeholder="tracking_results/manual_analysis.json",
            info="Если не указан, будет создан автоматически"
        )

        manual_run_btn = gr.Button("🚀 Запустить анализ (3 параллельных запроса)", variant="primary", size="lg")
        
    
    with gr.Tab("📊 Режим датасета"):
        with gr.Row():
            dataset_file = gr.File(
                label="📁 Файл датасета (TSV)",
                file_types=[".tsv"]
            )
            corpus_folder = gr.Textbox(
                label="📂 Папка с корпусом текстов",
                value="C:/Users/Admin/Documents/Thesis/corpus/annotated_texts",
                interactive=True
            )
        
        # Папка стартовых узлов
        start_nodes_folder = gr.Textbox(
            label="📁 Папка стартовых узлов (JSON)",
            value="examples/yandex-gpt5_candidates.json",
            interactive=True
        )
        
        start_nodes_info = gr.Textbox(
            label="ℹ️ Информация о стартовых узлах",
            interactive=False
        )
        
        dataset_info = gr.Textbox(
            label="ℹ️ Информация о датасете",
            interactive=False
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                word_dropdown = gr.Dropdown(
                    label="🎯 Поиск и выбор слова",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                    info="Начните печатать для поиска слов в датасете"
                )
            with gr.Column(scale=1):
                word_validation = gr.Textbox(
                    label="✓ Проверка слова",
                    interactive=False,
                    lines=2
                )
        
        sample_start = gr.Slider(
            minimum=1,
            maximum=1,
            value=1,
            step=1,
            label="🔢 Начальный индекс (для батч-режима)",
            interactive=False
        )
        num_samples = gr.Slider(
            minimum=1,
            maximum=1,
            value=1,
            step=1,
            label="🔢 Число примеров (для батч-режима)",
            interactive=False
        )
        dataset_size = gr.State(1)
        
        # Текст для выбранного слова
        word_text_display = gr.Textbox(
            label="📝 Текст для выбранного слова",
            lines=8,
            interactive=False
        )

        with gr.Row():
            prev_text_btn = gr.Button("←", variant="secondary", interactive=False)
            current_text_label = gr.Markdown(value='0', elem_classes="center-text")
            next_text_btn = gr.Button("→", variant="secondary", interactive=False)
        
        # Output file for dataset tracking
        dataset_output_file = gr.Textbox(
            label="💾 Файл для сохранения отслеживания датасета (опционально)",
            placeholder="tracking_results/dataset_analysis.json",
            info="Если не указан, будет создан автоматически"
        )
        
        
        # Батч-параметры
        with gr.Accordion("⚙️ Параметры батч-обработки", open=False):
            batch_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="📦 Размер батча",
                interactive=True
            )
            parallel_mode = gr.Radio(
                choices=["по текстам", "по умолчанию"],
                label="🔘 Режим параллелизма",
                value="по текстам",
                interactive=True
            )

        with gr.Row():
            dataset_run_btn = gr.Button("🚀 Обработать выбранное слово", variant="primary")
            batch_run_btn = gr.Button("🔄 Батч-обработка датасета", variant="secondary")
        
        batch_results = gr.Textbox(
            label="📊 Результаты батч-обработки",
            lines=5,
            interactive=False
        )
    
    # 3 parallel outputs
    gr.Markdown("### 📊 Параллельные процессы анализа")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🔵 Процесс #1")
            process_output_1 = gr.Textbox(
                label="Лог процесса #1",
                lines=15,
                max_lines=20,
                interactive=False
            )
            result_output_1 = gr.Textbox(
                label="Результат #1",
                lines=5,
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("#### 🟢 Процесс #2")
            process_output_2 = gr.Textbox(
                label="Лог процесса #2",
                lines=15,
                max_lines=20,
                interactive=False
            )
            result_output_2 = gr.Textbox(
                label="Результат #2",
                lines=5,
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("#### 🟡 Процесс #3")
            process_output_3 = gr.Textbox(
                label="Лог процесса #3",
                lines=15,
                max_lines=20,
                interactive=False
            )
            result_output_3 = gr.Textbox(
                label="Результат #3",
                lines=5,
                interactive=False
            )
    
    # Examples
    gr.Markdown("### 📝 Примеры")
    gr.Examples(
        examples=[
            [example_text, 50, 0.5, 0.95, False, ["get_hyponyms"]],
            ["Этот <predict_kb>велосипед</predict_kb> был изготовлен в Германии.", 50, 0.5, 0.95, True, ["get_hyponyms", "get_hypernyms"]],
            ["Новый <predict_kb>смартфон</predict_kb> имеет отличную камеру.", 50, 0.5, 0.95, False, ["get_hyponyms"]],
        ],
        inputs=[text_input, max_iterations, temperature, top_p, reranking, functions],
        label="Кликните на пример для загрузки"
    )
    
    # Info
    gr.Markdown("""
    ---
    ### ℹ️ Информация
    
    **Новые возможности:**
    1. **Отслеживание**
        - 📊 Все выбираемые синсеты автоматически сохраняются
        - 💾 Результаты сохраняются в JSON файлы в папке `tracking_results/`
        - 📈 Финальный результат содержит статистику по выбранным узлам
        - 🔍 Каждый вызов функции с node_id записывается в историю
    2. **Режим датасета**
        - Загрузка датасета с целевыми словами и путями до текстовых файлов из корпуса
        - Подгрузить стартовые узлы и настроить количество процессов и стартовых узлов из начала списка
        - Две возможности: обработка выбранного слова и батч-обработка
        - При выборе слова: выбор текста, если доступно несколько
        - В батч-режиме: настроить диапазон примеров, размер батча и тип параллелизма
        - Три типа параллелизма: по текстам (1 текст = 1 значение слова), по стартовым узлам (если они загружены) и стандартный (сэмплинг)
    
    **Параметры пайплайна:**
    - **Переранжирование** - сокращение числа синсетов с учётом релевантности к целевому слову
    - **Семантический анализ** - извлечение значения из текста для контекстуализированного переранжирования
    - **Функции** - выберите доступные функции для модели (get_hyponyms, get_hypernyms)
    
    **Возможные результаты:**
    - `not_found` - понятие не найдено в таксономии
    - `include in {synset_id}` - понятие является синонимом существующего синсета
    - `hyponym of {synset_id}` - понятие является гипонимом (более конкретным типом) существующего понятия
    
    **Требования:**
    - Запустите API: `python api.py`
    """)
    
    # Function to run 3 parallel processes using threads
    def run_parallel_analysis(target_word, texts, max_iterations, temperature, top_p, reranking, interpreting, functions, 
                         output_file, num_processes, start_nodes_path, max_n_starting_nodes, parallel_mode):
        """Run analysis with configurable number of processes"""
        
        # Загружаем стартовые узлы если указан файл и количество узлов > 0
        start_nodes_dict = {}
        if start_nodes_path and max_n_starting_nodes > 0:
            start_nodes_dict = load_start_nodes(start_nodes_path)
        
        # Извлекаем целевое слово из текста
        if not target_word:
            match = re.search(r'<predict_kb>(.*?)</predict_kb>', texts[0])
            target_word = match.group(1).strip() if match else None
        
        # Получаем стартовые узлы для целевого слова
        word_start_nodes = []
        if target_word and target_word in start_nodes_dict and max_n_starting_nodes > 0:
            word_start_nodes = start_nodes_dict[target_word][:max_n_starting_nodes]
        elif max_n_starting_nodes > 0:
            logger.warning(f'Word {target_word} not found in start nodes list ({list(start_nodes_dict.keys())[0]},...)')
        
        # Подготавливаем тексты/узлы для каждого процесса
        process_data = []
        for i in range(num_processes):
            if parallel_mode == 'по текстам' and i < len(texts):
                text = texts[i]
            else:
                text = texts[0]
            if word_start_nodes:
                # Используем предзаданный стартовый узел
                if parallel_mode == 'по стартовым узлам' and i < len(word_start_nodes):
                    start_node = word_start_nodes[i]
                else:
                    start_node = word_start_nodes[0]
                process_data.append({
                    'text': text,
                    'start_node_id': start_node,
                    'process_id': i + 1
                })
            else:
                # Стандартный режим без предзаданного узла
                process_data.append({
                    'text': text,
                    'process_id': i + 1
                })
        
        # Генерируем уникальные файлы для отслеживания
        output_files = [None] * num_processes
        timestamp = int(time.time())
        if output_file:
            base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
            extension = output_file.rsplit('.', 1)[1] if '.' in output_file else 'json'
            for i in range(num_processes):
                output_files[i] = f"{base_name}_process{i+1}_{timestamp}.{extension}"
        else:
            for i in range(num_processes):
                output_files[i] = f"tracking_results/single_word_process{i+1}_{timestamp}.json"
        
        # Создаем очереди для каждого активного процесса
        queues = [Queue() for _ in range(num_processes)]
        
        def run_stream_with_start_node(queue_idx, proc_data, max_iterations, temperature, top_p, 
                                    reranking, interpreting, functions, output_file):
            """Run streaming in a thread with optional start node"""
            start_node_id = proc_data.get('start_node_id')
            request_args = [proc_data['text'], max_iterations, temperature, top_p, reranking, interpreting, functions, output_file]
            if start_node_id:
                request_args.append(start_node_id)
            try:
                for process_log, final_result in process_text_stream(*request_args):
                    queues[queue_idx].put((process_log, final_result))
            except Exception as e:
                queues[queue_idx].put((f"❌ Ошибка в процессе #{proc_data['process_id']}: {str(e)}", ""))
            finally:
                queues[queue_idx].put(None)
        
        # Запускаем потоки
        threads = []
        for i in range(num_processes):
            thread = threading.Thread(
                target=run_stream_with_start_node,
                args=(i, process_data[i], max_iterations, temperature, top_p, reranking, interpreting, functions, output_files[i])
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Инициализируем результаты для всех 3 выходов
        results = [
            (f"🔵 Процесс #1 {'(стартовый узел: ' + process_data[0].get('start_node_id', 'нет') + ')' if len(process_data) > 0 else ''}", "") if num_processes >= 1 else ("Процесс отключен", ""),
            (f"🟢 Процесс #2 {'(стартовый узел: ' + process_data[1].get('start_node_id', 'нет') + ')' if len(process_data) > 1 else ''}", "") if num_processes >= 2 else ("Процесс отключен", ""),
            (f"🟡 Процесс #3 {'(стартовый узел: ' + process_data[2].get('start_node_id', 'нет') + ')' if len(process_data) > 2 else ''}", "") if num_processes >= 3 else ("Процесс отключен", "")
        ]
        
        active = [i < num_processes for i in range(3)]
        
        # Обновление результатов из очередей
        while any(active[:num_processes]):
            for i in range(num_processes):
                if active[i]:
                    try:
                        while not queues[i].empty():
                            item = queues[i].get_nowait()
                            if item is None:
                                active[i] = False
                            else:
                                results[i] = item
                    except:
                        pass
            
            yield (results[0][0], results[0][1],
                results[1][0], results[1][1],
                results[2][0], results[2][1])
            
            time.sleep(0.05)
        
        # Финальный результат
        yield (results[0][0], results[0][1],
            results[1][0], results[1][1],
            results[2][0], results[2][1])
        
    def run_manual_parallel_analysis(*args):
        if not isinstance(args[1], list):
            args[1] = [args[1]]
        return run_parallel_analysis(*args)

    # Event handlers
    manual_run_btn.click(
        fn=run_manual_parallel_analysis,
        inputs=[word_dropdown, text_input, max_iterations, temperature, top_p, reranking, interpreting, functions, 
                manual_output_file, num_processes, start_nodes_folder, max_n_starting_nodes],
        outputs=[process_output_1, result_output_1,
                process_output_2, result_output_2,
                process_output_3, result_output_3]
    )
    
    start_nodes_folder.change(
        fn=get_start_nodes_info,
        inputs=[start_nodes_folder],
        outputs=[start_nodes_info, max_n_starting_nodes, parallel_mode]
    )
    # Обновление слайдера максимального количества примеров
    def safe_get_dataset_info(file):
        if not file:
            return "Файл не выбран", gr.update(choices=[]), 1, 1, 1
        
        file_path = file.name if hasattr(file, 'name') else str(file)
        return get_dataset_info(file_path)

    dataset_file.change(
        fn=safe_get_dataset_info,
        inputs=[dataset_file],
        outputs=[dataset_info, word_dropdown, sample_start, num_samples, dataset_size]
    )
    sample_start.change(
        fn=lambda x, y: gr.update(maximum=y-x+1, value=y-x+1, interactive=True),
        inputs=[sample_start, dataset_size],
        outputs=[num_samples]
    )
    # Автозагрузка текста при выборе слова
    word_dropdown.change(
        fn=lambda dataset_path, corpus, word: load_word_text_from_corpus(dataset_path, corpus, word),
        inputs=[dataset_file, corpus_folder, word_dropdown],
        outputs=[word_text_display, prev_text_btn, current_text_label, next_text_btn]
    )
    # Выбор текста
    prev_text_btn.click(
        fn=lambda dataset_path, corpus, word, index: load_word_text_from_corpus(dataset_path, corpus, word, int(index) - 2),
        inputs=[dataset_file, corpus_folder, word_dropdown, current_text_label],
        outputs=[word_text_display, prev_text_btn, current_text_label, next_text_btn]
    )
    next_text_btn.click(
        fn=lambda dataset_path, corpus, word, index: load_word_text_from_corpus(dataset_path, corpus, word, int(index)),
        inputs=[dataset_file, corpus_folder, word_dropdown, current_text_label],
        outputs=[word_text_display, prev_text_btn, current_text_label, next_text_btn]
    )
    reranking.change(
        fn=lambda x: gr.update(interactive=False) if not x else gr.update(interactive=True),
        inputs=[reranking],
        outputs=[interpreting]
    )
    # Режим выбранного слова
    dataset_run_btn.click(
        fn=process_dataset_item,
        inputs=[
            dataset_file, corpus_folder, word_dropdown, max_iterations,
            temperature, top_p, reranking, interpreting, functions,
            num_processes, start_nodes_folder, max_n_starting_nodes,
            parallel_mode, dataset_output_file
        ],
        outputs=[process_output_1, result_output_1,
                process_output_2, result_output_2,
                process_output_3, result_output_3]
    )
    word_dropdown.change(
        fn=lambda dataset_file, query: search_words_in_dataset(dataset_file.name if dataset_file else "", query) if query else query.upper(),
        inputs=[dataset_file, word_dropdown],
        outputs=[word_dropdown]
    )
    
    # Валидация выбранного слова
    word_dropdown.select(
        fn=lambda dataset_file, word: validate_word_in_dataset(dataset_file.name if dataset_file else "", word),
        inputs=[dataset_file, word_dropdown],
        outputs=[word_validation]
    )
    
    # Загрузка текста при валидном выборе слова
    word_dropdown.select(
        fn=lambda dataset_path, corpus, word: load_word_text_from_corpus(dataset_path, corpus, word),
        inputs=[dataset_file, corpus_folder, word_dropdown],
        outputs=[word_text_display, prev_text_btn, current_text_label, next_text_btn]
    )
    
    # Батч-обработка с учетом max_samples
    batch_run_btn.click(
        fn=process_dataset_batch,
        inputs=[dataset_file, corpus_folder, sample_start, num_samples, num_processes, batch_size, max_iterations, 
                temperature, top_p, reranking, interpreting, functions, start_nodes_folder, max_n_starting_nodes, parallel_mode],
        outputs=[batch_results]
    )
    

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5003, share=False)