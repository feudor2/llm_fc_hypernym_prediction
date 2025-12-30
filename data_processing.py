import json
import os
import re
from pathlib import Path
from taxoenrich.data_utils import read_dataset


def load_dataset(dataset_path):
    """
    Функция считывает датасет в формате {"word_1": [["node_id_1", "node_id_2",...]], ...}
    где word_i — это целевое слово, для которого нужно предсказать позицию
    node_id_i — узел, подходящий для использования в качестве гиперонима (целевой родительский узел)
    """
    try:
        dataset = read_dataset(dataset_path, read_fn=json.loads)
        return dataset
    except Exception as e:
        raise Exception(f"Ошибка загрузки датасета: {str(e)}")


def extract_context_around_tag(text, context_sentences=2):
    """
    Извлекает контекст вокруг тега <predict_kb>
    
    Args:
        text: полный текст
        context_sentences: количество предложений с каждой стороны
    
    Returns:
        Сокращенный текст с контекстом
    """
    # Найти позицию тега в тексте
    tag_pattern = f"<predict_kb>.*</predict_kb>"
    match = re.search(tag_pattern, text)
    
    if not match:
        return text  # Если тег не найден, вернуть весь текст
    
    tag_start = match.start()
    tag_end = match.end()
    
    # Разбить текст на предложения
    sentences = re.split(r'[.!?]+', text)
    
    # Найти предложение с тегом
    current_pos = 0
    tag_sentence_idx = -1
    
    for i, sentence in enumerate(sentences):
        sentence_end = current_pos + len(sentence) + 1  # +1 для разделителя
        if current_pos <= tag_start < sentence_end:
            tag_sentence_idx = i
            break
        current_pos = sentence_end
    
    if tag_sentence_idx == -1:
        return text  # Если не удалось найти предложение, вернуть весь текст
    
    # Определить границы контекста
    start_idx = max(0, tag_sentence_idx - context_sentences)
    end_idx = min(len(sentences), tag_sentence_idx + context_sentences + 1)
    
    # Собрать контекст
    context_sentences_list = sentences[start_idx:end_idx]
    
    # Добавить многоточие если текст был сокращен
    result_parts = []
    
    if start_idx > 0:
        result_parts.append("...")
    
    result_parts.extend(context_sentences_list)
    
    if end_idx < len(sentences):
        result_parts.append("...")
    
    return '. '.join(result_parts).strip()


def load_corpus_text(corpus_folder, word):
    """
    Функция чтения файлов из корпуса:
    - извлекается первый файл с префиксом, соответствующим слову
    - извлекается содержимое тега <predict_kb>...</predict_kb>
    - сокращается до контекста вокруг тега
    
    Args:
        corpus_folder: путь к папке с корпусом
        word: целевое слово для поиска
    
    Returns:
        Текст с тегом или None если не найден
    """
    if not corpus_folder or not os.path.exists(corpus_folder):
        return None
    
    corpus_path = Path(corpus_folder)
    
    # Найти файл с префиксом слова
    matching_files = []
    for file_path in corpus_path.glob(f"{word}*"):
        if file_path.is_file() and file_path.suffix in ['.txt', '.text', '']:
            matching_files.append(file_path)
    
    if not matching_files:
        return None
    
    # Взять первый подходящий файл
    file_path = matching_files[0]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Найти тег <predict_kb>
        tag_match = re.search(r'<predict_kb>(.*?)</predict_kb>', content)
        if not tag_match:
            return None
        
        tag_content = tag_match.group(1)
        
        # Сократить текст до контекста вокруг тега
        context_text = extract_context_around_tag(content, tag_content)
        
        return context_text
        
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {str(e)}")
        return None


def get_available_words(corpus_folder):
    """
    Получить список доступных слов в корпусе
    
    Args:
        corpus_folder: путь к папке с корпусом
    
    Returns:
        Список имен файлов (без расширений)
    """
    if not corpus_folder or not os.path.exists(corpus_folder):
        return []
    
    corpus_path = Path(corpus_folder)
    words = []
    
    for file_path in corpus_path.glob("*"):
        if file_path.is_file():
            # Извлечь имя без расширения как потенциальное слово
            word = file_path.stem
            words.append(word)
    
    return sorted(set(words))