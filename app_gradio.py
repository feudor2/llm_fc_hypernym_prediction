import gradio as gr
import requests
import json
from typing import Generator
import threading
from queue import Queue
import time
import os
from pathlib import Path

# Импортируем функции для работы с данными
from data_processing import load_dataset, load_corpus_text


def process_text_stream(text: str, max_iterations: int, temperature: float, top_p: float, 
                       reranking: bool, functions: list):
    """Process text using the streaming API endpoint"""
    
    # Validate input
    if '<predict_kb>' not in text or '</predict_kb>' not in text:
        yield "❌ Ошибка: Текст должен содержать теги <predict_kb>...</predict_kb>", ""
        return
    
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
        "functions": functions
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
    """Get information about loaded dataset"""
    if not dataset_path or not os.path.exists(dataset_path):
        return "Датасет не выбран или файл не найден", [], ""
    
    try:
        dataset = load_dataset(dataset_path)
        words = list(dataset.keys())
        info = f"📊 Загружен датасет: {len(words)} слов"
        return info, words, words[0] if words else ""
    except Exception as e:
        return f"❌ Ошибка загрузки датасета: {str(e)}", [], ""


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
                placeholder="Выберите папку с текстовыми файлами",
                interactive=True
            )
        
        # Browse buttons for file explorer access
        with gr.Row():
            browse_dataset_btn = gr.Button("🔍 Обзор датасета", size="sm")
            browse_corpus_btn = gr.Button("🔍 Обзор корпуса", size="sm")
        
        dataset_info = gr.Textbox(
            label="ℹ️ Информация о датасете",
            interactive=False
        )
        
        with gr.Row():
            word_dropdown = gr.Dropdown(
                label="🎯 Выберите слово",
                choices=[],
                interactive=True
            )
            num_samples = gr.Slider(
                minimum=1,
                maximum=100,
                value=1,
                step=1,
                label="🔢 Количество примеров"
            )
        
        dataset_run_btn = gr.Button("🚀 Обработать выбранное слово", variant="primary", size="lg")
    
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
                        ("get_hyponyms", "get_hyponyms"),
                        ("get_hypernyms", "get_hypernyms")
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
    
    dataset_run_btn.click(
        fn=lambda dataset_file, corpus_folder, word, max_iter, temp, top_p_val, rerank, funcs: 
            list(process_dataset_item(
                dataset_file.name if dataset_file else "", 
                corpus_folder, word, max_iter, temp, top_p_val, rerank, funcs
            ))[-1] if dataset_file and corpus_folder and word else ("❌ Заполните все поля", ""),
        inputs=[dataset_file, corpus_folder, word_dropdown, max_iterations, temperature, top_p, reranking, functions],
        outputs=[process_output_1, result_output_1]
    )
    
    # Update dataset info when file is uploaded
    dataset_file.change(
        fn=lambda file: get_dataset_info(file.name if file else ""),
        inputs=[dataset_file],
        outputs=[dataset_info, word_dropdown, word_dropdown]
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=5003, share=False)