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

# Импортируем функции для работы с данными
from data_processing import load_dataset, load_corpus_text

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_text_stream(text: str, max_iterations: int, temperature: float, top_p: float, 
                       reranking: bool, functions: list):
    """Process text using the streaming API endpoint"""
    
    # Validate input
    if '<predict_kb>' not in text or '</predict_kb>' not in text:
        yield "❌ Ошибка: Текст должен содержать теги <predict_kb>...</predict_kb>", ""
        return
    
    # Исправить координацию с tools.py
    valid_functions = ["get_hyponyms", "get_hypernyms"]
    functions = [f for f in functions if f in valid_functions]
    
    if not functions:
        functions = ["get_hyponyms"]  # Fallback
    
    # API URL (hardcoded)
    api_url = "http://localhost:8500"
    
    # Prepare request with pipeline parameters
    endpoint = f"{api_url.rstrip('/')}/predict/stream"
    payload = {
        "text": text,
        "max_iterations": max_iterations,
        "temperature": temperature,
        "top_p": top_p,
        "reranking": reranking,
        "functions": functions  # Теперь правильно передается
    }
    
    process_log = ""
    final_result = ""
    
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
                            
                            elif event_type == 'final':
                                final_result = f"✅ Финальный результат:\n\n{data['result']}"
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


def process_dataset_item(dataset_path: str, corpus_folder: str, word: str, 
                        max_iterations: int, temperature: float, top_p: float, 
                        reranking: bool, functions: list):
    """Process a specific word from dataset using corpus text"""
    try:
        # Load corpus text for the word
        text = load_corpus_text(corpus_folder, word)
        if not text:
            yield f"❌ Не найден текст для слова: {word}", ""
            return
        
        # Process using the streaming function
        for process_log, final_result in process_text_stream(
            text, max_iterations, temperature, top_p, reranking, functions
        ):
            yield process_log, final_result
    except Exception as e:
        yield f"❌ Ошибка при обработке слова '{word}': {str(e)}", ""


def get_dataset_info(dataset_path: str):
    if not dataset_path or not os.path.exists(dataset_path):
        return "Датасет не выбран или файл не найден", gr.update(choices=[], interactive=False), gr.update(maximum=1, value=1)
    
    try:
        dataset = load_dataset(dataset_path)
        words = list(dataset.keys())
        max_samples = len(words)
        info = f"📊 Загружен датасет: {len(words)} слов"
        
        # Возвращаем пустые choices для dropdown, но делаем его интерактивным для поиска
        return (
            info, 
            gr.update(choices=[], interactive=True, allow_custom_value=True),
            gr.update(maximum=max_samples, value=max_samples, interactive=True)
        )
    except Exception as e:
        return f"❌ Ошибка загрузки датасета: {str(e)}", gr.update(choices=[], interactive=False), gr.update(maximum=1, value=1)

def search_words_in_dataset(dataset_path: str, search_query: str):
    """Search for words in dataset that match the query"""
    if not dataset_path or not search_query:
        return gr.update(choices=[])
    
    try:
        dataset = load_dataset(dataset_path)
        words = list(dataset.keys())
        
        # Поиск слов, содержащих запрос (регистронезависимо)
        search_query = search_query.upper().strip()
        matching_words = [word for word in words if search_query in word.upper()]
        
        # Ограничиваем результат до 50 слов для производительности
        matching_words = matching_words[:50]
        
        return gr.update(choices=matching_words)
    except Exception as e:
        logger.error(f"Ошибка поиска в датасете: {e}")
        return gr.update(choices=[])

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

# Добавить функцию загрузки текста из корпуса:
def load_word_text_from_corpus(corpus_folder: str, word: str):
    """Load text for specific word from corpus"""
    if not corpus_folder or not word:
        return ""
    
    # Ищем файл СЛОВО.txt
    file_path = os.path.join(corpus_folder, f"{word}.txt")
    logger.debug(f'Ищем файл {file_path}')
    
    if os.path.exists(file_path):
        try:
            text = load_corpus_text(corpus_folder, word)
            logger.debug(f'Извлечённый текст {text}')
            return text
        except Exception as e:
            return f"❌ Ошибка чтения файла: {str(e)}"
    
    return f"❌ Файл {word}.txt не найден в корпусе"

# Добавить функцию батч-обработки:
def process_dataset_batch(dataset_file, corpus_folder, max_samples, batch_size, max_iterations, 
                         temperature, top_p, reranking, functions, progress=gr.Progress()):
    """Process dataset in batches with max_samples limit"""
    if not dataset_file or not corpus_folder:
        return "❌ Выберите датасет и папку корпуса"
    
    try:
        dataset = load_dataset(dataset_file.name)
        all_words = list(dataset.keys())
        
        # Ограничиваем количество слов параметром max_samples
        words_to_process = all_words[:max_samples]
        total_words = len(words_to_process)
        
        logger.info(f"Обработка {total_words} слов из {len(all_words)} (ограничение: {max_samples})")
        
        results = {}
        processed_count = 0
        
        for i in range(0, total_words, batch_size):
            batch_words = words_to_process[i:i+batch_size]
            batch_end = min(i + batch_size, total_words)
            
            progress((processed_count)/total_words, f"Обработка слов {i+1}-{batch_end} из {total_words}")
            
            # Обрабатываем каждое слово в батче
            for word in batch_words:
                try:
                    text = load_word_text_from_corpus(corpus_folder, word)
                    if "❌" in text:
                        results[word] = {"error": text}
                        processed_count += 1
                        continue
                    
                    # Получаем последний результат из генератора
                    stream_results = list(process_text_stream(
                        text, max_iterations, temperature, top_p, reranking, functions
                    ))
                    if stream_results:
                        final_log, final_result = stream_results[-1]
                        results[word] = {
                            "result": final_result,
                            "log": final_log
                        }
                    else:
                        results[word] = {"error": "Нет результатов"}
                        
                except Exception as e:
                    logger.error(f"Ошибка обработки слова {word}: {e}")
                    results[word] = {"error": str(e)}
                
                processed_count += 1
                progress(processed_count/total_words, f"Обработано {processed_count}/{total_words} слов")
        
        # Сохраняем результаты
        output_file = f"test_results/batch_results_{total_words}words.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return f"✅ Обработка завершена. Обработано {processed_count} слов из {len(all_words)}. Результаты сохранены в {output_file}"
        
    except Exception as e:
        logger.error(f"Ошибка батч-обработки: {e}")
        return f"❌ Ошибка батч-обработки: {str(e)}"
    
# Добавить функцию для стартовых узлов:
def get_start_nodes_info(start_nodes_folder: str):
    """Get information about start nodes folder"""
    if not start_nodes_folder or not os.path.exists(start_nodes_folder):
        return "Папка стартовых узлов не выбрана или не найдена"
    
    json_files = glob.glob(os.path.join(start_nodes_folder, "*.json"))
    return f"📁 Найдено {len(json_files)} файлов стартовых узлов"


# Example text
example_text = '''Каждое лето группы энтузиастов испытывают себя и отправляются на поиски снега и льда. Чаще всего их называют альпинисты, и они в любое время года не против пересечь ледник или тропить по снегу до вершины. Храбрые профи даже готовы лезть по скалам со льдом, выбирая запредельно сложные маршруты. Горные туристы тоже с удовольствием гуляют среди вечной мерзлоты на высотах более 4000 метров над уровнем моря. И всем им требуется надёжное сцепление на скользкой поверхности льда.

Итальянский кузнец и основатель легендарной альпинистской компании Генри Гривель более 100 лет назад снабдил одних из первых восходителей прообразом того, что сейчас называют <predict_kb>кошками</predict_kb>. Устройства были больше похожи на ряд соединённых скоб с заострёнными шипами и ремнями для крепления. Они изменили тактику передвижения по снежно-ледовому склону и значительно расширили возможности спортсменов.

С тех времён модели заметно усовершенствовали, но по-прежнему это изделия из металла, которые крепятся к ботинкам, вгрызаются в лёд и держат на снежном рельефе'''


# Create Gradio interface
with gr.Blocks(title="RuWordNet Taxonomy Prediction Client", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 RuWordNet Taxonomy Prediction Client")
    gr.Markdown("""
    Этот интерфейс обращается к API сервису для анализа текста и определения места понятия в таксономии RuWordNet.
    
    **Инструкция:**
    1. Выберите режим работы: "Ручной ввод" или "Датасет"
    2. Настройте параметры модели и пайплайна
    3. Введите текст или выберите файлы
    4. Запустите анализ
    """)
    
    # Mode selection
    with gr.Tab("🖊️ Ручной ввод"):
        # Text input
        text_input = gr.Textbox(
            label="📝 Входной текст",
            placeholder="Введите текст с тегами <predict_kb>...</predict_kb>",
            lines=10,
            value=example_text
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
                value="corpus/private",
                interactive=True
            )
        
        # Папка стартовых узлов
        start_nodes_folder = gr.Textbox(
            label="📁 Папка стартовых узлов (JSON)",
            value="examples",
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
        
        num_samples = gr.Slider(
            minimum=1,
            maximum=1,  # Будет обновляться динамически
            value=1,
            step=1,
            label="🔢 Максимум примеров (для батч-режима)"
        )
        
        # Текст для выбранного слова
        word_text_display = gr.Textbox(
            label="📝 Текст для выбранного слова",
            lines=8,
            interactive=False
        )
        
        with gr.Row():
            dataset_run_btn = gr.Button("🚀 Обработать выбранное слово", variant="primary")
            batch_run_btn = gr.Button("🔄 Батч-обработка датасета", variant="secondary")
        
        # Батч-параметры
        with gr.Accordion("⚙️ Параметры батч-обработки", open=False):
            batch_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Размер батча"
            )
        
        batch_results = gr.Textbox(
            label="📊 Результаты батч-обработки",
            lines=5,
            interactive=False
        )
    
    # Parameters section (shared)
    with gr.Accordion("⚙️ Параметры", open=True):
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
                reranking = gr.Checkbox(
                    label="🔄 Переранжирование",
                    value=False
                )
                functions = gr.CheckboxGroup(
                    label="🔧 Функции",
                    choices=[
                        ("Получить гипонимы", "get_hyponyms"),
                        ("Получить гиперонимы", "get_hypernyms")
                    ],
                    value=["get_hyponyms"]
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
                lines=3,
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
                lines=3,
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
                lines=3,
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
    
    **Параметры пайплайна:**
    - **Переранжирование** - включает дополнительную обработку результатов
    - **Функции** - выберите доступные функции для модели (get_hyponyms, get_hypernyms)
    
    **Возможные результаты:**
    - `not_found` - понятие не найдено в таксономии
    - `include in {synset_id}` - понятие является синонимом существующего синсета
    - `hyponym of {synset_id}` - понятие является гипонимом (более конкретным типом) существующего понятия
    
    **Требования:**
    - Запустите API: `python api.py`
    """)
    
    # Function to run 3 parallel processes using threads
    def run_parallel_analysis(text, max_iterations, temperature, top_p, reranking, functions):
        """Run 3 parallel analysis processes using threads"""
        
        # Queues to communicate between threads
        queues = [Queue(), Queue(), Queue()]
        
        def run_stream(queue_idx, text, max_iterations, temperature, top_p, reranking, functions):
            """Run streaming in a thread and put results in queue"""
            try:
                for process_log, final_result in process_text_stream(
                    text, max_iterations, temperature, top_p, reranking, functions
                ):
                    queues[queue_idx].put((process_log, final_result))
            except Exception as e:
                queues[queue_idx].put((f"❌ Ошибка в потоке: {str(e)}", ""))
            finally:
                # Signal completion
                queues[queue_idx].put(None)
        
        # Start 3 threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=run_stream,
                args=(i, text, max_iterations, temperature, top_p, reranking, functions)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Track state for each process
        active = [True, True, True]
        results = [("🔵 Запуск процесса #1...", ""), ("🟢 Запуск процесса #2...", ""), ("🟡 Запуск процесса #3...", "")]
        
        # Continuously check queues and yield updates
        while any(active):
            updated = False
            
            for i in range(3):
                if active[i]:
                    try:
                        # Try to get item without blocking
                        while not queues[i].empty():
                            item = queues[i].get_nowait()
                            if item is None:
                                active[i] = False
                            else:
                                results[i] = item
                                updated = True
                    except:
                        pass
            
            # Yield current state (even if not updated to keep UI responsive)
            yield (results[0][0], results[0][1],
                   results[1][0], results[1][1],
                   results[2][0], results[2][1])
            
            # Small sleep to prevent busy waiting but keep responsive
            time.sleep(0.05)
        
        # Final yield to ensure all results are shown
        yield (results[0][0], results[0][1],
               results[1][0], results[1][1],
               results[2][0], results[2][1])
    
    # Event handlers
    manual_run_btn.click(
        fn=run_parallel_analysis,
        inputs=[text_input, max_iterations, temperature, top_p, reranking, functions],
        outputs=[process_output_1, result_output_1,
                 process_output_2, result_output_2,
                 process_output_3, result_output_3]
    )
    start_nodes_folder.change(
        fn=get_start_nodes_info,
        inputs=[start_nodes_folder],
        outputs=[start_nodes_info]
    )
    # Обновление слайдера максимального количества примеров
    def safe_get_dataset_info(file):
        if not file:
            return "Файл не выбран", gr.update(choices=[]), 1
        
        file_path = file.name if hasattr(file, 'name') else str(file)
        return get_dataset_info(file_path)

    dataset_file.change(
        fn=safe_get_dataset_info,
        inputs=[dataset_file],
        outputs=[dataset_info, word_dropdown, num_samples]
    )
    # Автозагрузка текста при выборе слова
    word_dropdown.change(
        fn=lambda word, corpus: load_word_text_from_corpus(corpus, word),
        inputs=[word_dropdown, corpus_folder],
        outputs=[word_text_display]
    )
    # Режим выбранного слова
    dataset_run_btn.click(
        fn=process_dataset_item,
        inputs=[dataset_file, corpus_folder, word_dropdown, max_iterations, temperature, top_p, reranking, functions],
        outputs=[process_output_1, result_output_1]
    )
    word_dropdown.change(
        fn=lambda dataset_file, query: search_words_in_dataset(dataset_file.name if dataset_file else "", query) if query else gr.update(),
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
        fn=lambda word, corpus: load_word_text_from_corpus(corpus, word),
        inputs=[word_dropdown, corpus_folder],
        outputs=[word_text_display]
    )
    
    # Батч-обработка с учетом max_samples
    batch_run_btn.click(
        fn=process_dataset_batch,
        inputs=[dataset_file, corpus_folder, num_samples, batch_size, max_iterations, 
                temperature, top_p, reranking, functions],
        outputs=[batch_results]
    )
    

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5003, share=False)