import os
import json
import asyncio
import logging
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from utils.utils import delayed_call
from utils.io_utils import aread_txt
from prompt_functions.context_analyser import ContextAnalyser, ContextAnalyserRequest

load_dotenv()

# Initialize client
oclient = AsyncOpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Initialize logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_dir = 'logs'

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
)
logger = logging.getLogger()
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)
handlers = [
    logging.StreamHandler(),
    logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y%m%d')}_context_analyser.log", encoding='utf8')
]
for handler in handlers:
    logger.addHandler(handler)



async def prepare_corpus(corpus_path: str, n_texts: 10):
    tasks = []
    filenames = os.listdir(corpus_path)[:]
    for i, filepath in enumerate(filenames):
        task = delayed_call(
            delay=i*0.01,
            func=aread_txt,
            args=(os.path.join(corpus_path, filepath),),
            kwargs={}
        )
        tasks.append(task)

    responses = await tqdm_asyncio.gather(*tasks, desc='Reading contexts')
    words = [filename.split('_')[0] for filename in filenames]
    output = defaultdict(list)
    for word, response, filename in zip(words, responses, filenames):
        output[word].append((response, filename))
    return output


async def run_test(context_analyser: ContextAnalyser, requests: list[ContextAnalyserRequest], batch_size: int = 1):
    async def analyse(request: ContextAnalyserRequest, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await context_analyser(request)

    semaphore = asyncio.Semaphore(batch_size)
    tasks = [analyse(request, semaphore) for request in requests]
    results = await tqdm_asyncio.gather(*tasks, desc='Running tests')
    return results 

async def prepare_senses(dataset_path: str) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    senses = defaultdict(list)
    lines = await aread_txt(dataset_path)
    for line in lines.split('\n'):
        try:
            word, node_ids, parents = line.split('\t')
            senses[word].append((eval(node_ids), eval(parents.strip())))
        except ValueError:
            continue
    return senses


class SenseTracker:
    def __init__(self, senses: Dict[str, List[Tuple[List[str], List[str]]]]):
        self.iteration = 0
        self.senses = {word: {i: [] for i in range(len(_senses))} for word, _senses in senses.items()}

    def select_texts(self, corpus: Dict[str, List[Tuple[str, str]]]) -> List[Tuple[str, Tuple[str, str]]]:
        contexts = []
        for word, _senses in self.senses.items():
            if not all(_senses.values()) and self.iteration < len(corpus[word]):
                contexts.append((word, corpus[word][self.iteration]))
        return contexts
        
    def register(self, results: List[str], contexts: List[Tuple[str, Tuple[str, str]]]):
        for index, (word, (context, filename)) in zip(results, contexts):
            if isinstance(index, int) or (isinstance(index, str) and index.isdigit()):
                index = int(index)
                if index in self.senses[word]:
                    self.senses[word][index].append(filename)

        self.iteration += 1

async def rebuild_dataset(tracker: SenseTracker, senses: Dict[str, List[Tuple[List[str], List[str]]]]):
    output = {'word': [], 'node_ids': [], 'senses': [], 'files': []}
    for word, index in tracker.senses.items():
        for i, filenames in index.items():
            if filenames:
                node_ids, parents = senses[word][i]
                output['word'].append(word)
                output['node_ids'].append(node_ids)
                output['senses'].append(parents)
                output['files'].append(filenames)

    return pd.DataFrame(output)

async def main():
    corpus_path = 'corpus/annotated_texts'
    corpus = await prepare_corpus(corpus_path, n_texts=10)
    dataset_path = 'datasets/nouns_private_subgraphs.tsv'
    senses = await prepare_senses(dataset_path)
    tracker = SenseTracker(senses)
    context_analyser = ContextAnalyser(oclient, model_name=os.environ['MODEL_NAME'], logger=logger)
    while True:
        contexts = tracker.select_texts(corpus)
        if not contexts:
            break
        requests = [ContextAnalyserRequest(context=pair[1][0], senses=[item[1] for item in senses[pair[0]]]) for pair in contexts]
        results = await run_test(context_analyser, requests, batch_size=10)
        tracker.register(results, contexts)

    results = await rebuild_dataset(tracker=tracker, senses=senses)
    results.to_csv('test_results/context_analyser_results.tsv', sep='\t', header=None, index=False)


if __name__ == "__main__":
    asyncio.run(main())