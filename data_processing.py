import json
import os
import re
import asyncio

from pathlib import Path
from functools import lru_cache
from collections import defaultdict

from pymorphy3 import MorphAnalyzer
#from taxoenrich.data_utils import read_dataset
from io_utils import aread_json

parser = MorphAnalyzer()

def read_dataset(dataset_path):
    """
    Обновленная функция чтения с добавление путей хранения файлов контекста
    
    :param dataset_path: путь до датаста
    """
    output = defaultdict(list)
    with open(dataset_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            word, node_ids, node_names, files = line.strip().split('\t')
            output[word].append((eval(node_ids), eval(files)))
    return output


def load_dataset(dataset_path):
    """
    Функция считывает датасет в формате {"word_1": [(["node_id_1_1", "node_id_1_2",...], ["path_1_1", "path_1_2", ...]]), ...]...}
    где word_i — это целевое слово, для которого нужно предсказать позицию
    node_id_i — узел, подходящий для использования в качестве гиперонима (целевой родительский узел)
    """
    try:
        dataset = read_dataset(dataset_path)
        return dataset
    except Exception as e:
        raise Exception(f"Ошибка загрузки датасета: {str(e)}")
    
def convert_paths(dataset: dict[str, list[tuple[list[str], list[str]]]], index: int = None):
    '''
    Функция преобразует датасет в формат {"word_1": ["path_1_1", "path_2_1", ...], ...}
    
    :param dataset: датасет
    :type dataset: dict[str, list[tuple[list[str], list[str]]]]
    :param index: индекс текста (все если None)
    :type index: int
    '''
    if index is not None:
        return {word: [sense[1][index] for sense in senses if index < len(sense[1])] for word, senses in dataset.items()}
    return {word: [path for sense in senses for path in sense[1]] for word, senses in dataset.items()}


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


def load_corpus_text(corpus_folder, item_path):
    """
    Функция чтения файлов из корпуса
    """
    if not (corpus_folder and item_path) or not os.path.exists(os.path.join(corpus_folder, item_path)):
        return None
    
    file_path = Path(os.path.join(corpus_folder, item_path))
    
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

def load_start_nodes(start_nodes_path, key_fn=lambda x: x.upper()):
    """
    Загружает стартовые узлы для слова из файла start_nodes_path
    
    Args:
        start_nodes_path: путь к файлу со стартовыми узлами    
    Returns:
        Список ID узлов или None если не найден
    """
    if not start_nodes_path or not os.path.exists(start_nodes_path):
        return {}
    try:
        with open(start_nodes_path, 'r', encoding='utf-8') as f:
            nodes = {key_fn(name): node_ids for name, node_ids in json.load(f).items()}
        return nodes
        
    except Exception as e:
        print(f"Ошибка чтения файла {start_nodes_path}: {str(e)}")
        return None



async def collect_tracking_results(tracking_path: str):
    def analyse_single_file(content, filename):
        word = content[0].get('target_word')
        synsets = content[0].get('selected_synsets')
        total_selections = content[0].get('total_selections')
        if synsets:
            synsets = [synset['synset_id'] for synset in synsets]
                
        return {
            'word': word,
            'filename': filename,
            'synsets': synsets,
            'total_selections': total_selections
        }

    tasks = []
    files = []
    with os.scandir(tracking_path) as entries:
        for entry in entries:
            tasks.append(aread_json(entry.path))
            files.append(entry.name)
    parsed = await asyncio.gather(*tasks)
    processed = [analyse_single_file(content, filename) for content, filename in zip(parsed, files)]
    return processed


@lru_cache(maxsize=10000)
def normalize_word(word: str) -> str:
    return parser.parse(word)[0].normal_form

def prepare_target(item: str) -> str:
    queries = [normalize_word(part) for part in item.lower().split(' ')]
    return ' '.join(queries)
    
if __name__ == "__main__":
    print(load_corpus_text('corpus/private', 'АБСЕНТЕИЗМ'))