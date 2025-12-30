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


def extract_context_around_tag(text, tag_content, context_sentences=2):
    """
    Извлекает контекст вокруг тега <predict_kb>
    """
    # Найти любой тег <predict_kb>...</predict_kb> в тексте
    tag_pattern = r'<predict_kb>(.*?)</predict_kb>'
    match = re.search(tag_pattern, text)
    
    if not match:
        print('Tag not found while extracting context')
        return text
    
    tag_start = match.start()
    tag_end = match.end()
    
    # Разбить текст на предложения (улучшенная регулярка)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Найти предложение с тегом
    current_pos = 0
    tag_sentence_idx = -1
    
    for i, sentence in enumerate(sentences):
        sentence_start = current_pos
        sentence_end = current_pos + len(sentence)
        
        if sentence_start <= tag_start < sentence_end:
            tag_sentence_idx = i
            break
        current_pos = sentence_end + 1  # +1 для пробела
    
    if tag_sentence_idx == -1:
        #print('Sentence with tag not found')
        return text
    
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
    
    result = ' '.join(result_parts).strip()
    
    # Добавить отладочный вывод
    #print(f"Extracted context ({len(result)} chars): {result[:200]}...")
    
    return result


def load_corpus_text(corpus_folder, word):
    """
    Функция чтения файлов из корпуса
    """
    if not corpus_folder or not os.path.exists(corpus_folder):
        return None
    
    corpus_path = Path(corpus_folder)
    
    # Найти файл с именем слова
    file_path = corpus_path / f"{word}.txt"
    
    if not file_path.exists():
        #print(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Найти тег <predict_kb>
        tag_match = re.search(r'<predict_kb>(.*?)</predict_kb>', content)
        if not tag_match:
            #print('Tag not found in file content')
            return None
        
        tag_content = tag_match.group(1)
        #print(f"Found tag content: '{tag_content}'")
        
        # Сократить текст до контекста вокруг тега
        context_text = extract_context_around_tag(content, tag_content, context_sentences=3)
        
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

def load_start_nodes(start_nodes_folder, word):
    """
    Загружает стартовые узлы для слова из файла СЛОВО.json
    
    Args:
        start_nodes_folder: путь к папке со стартовыми узлами
        word: целевое слово
    
    Returns:
        Список ID узлов или None если не найден
    """
    if not start_nodes_folder or not os.path.exists(start_nodes_folder):
        return None
    
    file_path = Path(start_nodes_folder) / f"{word}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        return nodes
        
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {str(e)}")
        return None
    
if __name__ == "__main__":
    print(load_corpus_text('corpus/private', 'АБСЕНТЕИЗМ'))