import os
import asyncio
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau, spearmanr


from utils.io_utils import aread_json
from prompt_functions.reranker import Reranker, RerankerRequest
from taxoenrich.core import RuWordNet

load_dotenv()

# Initialize client
oclient = OpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Initialize RuWordNet
wordnet = RuWordNet('wordnets/RuWordNet')

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
    logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y%m%d')}_reranker.log")
]
for handler in handlers:
    logger.addHandler(handler)


def calculate_ndcg(true, pred):
    y_true = np.asarray([true])
    y_pred = np.asarray([pred])
    return ndcg_score(y_true, y_pred)

def calculate_kendalltau(true, pred):
    correlation, p_value = kendalltau(true, pred)
    return correlation

def calculate_spearmanr(true, pred):
    correlation, p_value = spearmanr(true, pred)
    return correlation

def evaluate(dataset, results):
    words, items = list(dataset.keys()), list(dataset.values())
    true_ranks = [[float(r) / 5 for r, values in ranked.items() for candidate in values] for ranked in items]
    pred_ranks = [[item['score'] for item in result] for result in results]
    metrics = defaultdict(list)
    for word, true, pred in zip(words, true_ranks, pred_ranks):
        pred = np.pad(pred, (0, len(true) - len(pred)), mode='constant')
        metrics['word'].append(word)
        metrics['true'].append(true)
        metrics['pred'].append(pred)
        metrics['ndcg'].append(calculate_ndcg(true, pred))
        metrics['tau'].append(calculate_kendalltau(true, pred))
        metrics['r'].append(calculate_spearmanr(true, pred))
    return metrics


async def prepare_dataset(path: str):
    def replace_ids_by_names(ids: list[str]):
        return [wordnet.synsets[node_id].synset_name for node_id in ids]

    dataset = await aread_json(path)
    dataset = {word: {rank: replace_ids_by_names(ids) for rank, ids in ranks.items()} for word, ranks in list(dataset.items())}
    return dataset

def prepare_requests(dataset):
    def _merge(ranked):
        return [candidate for r, values in ranked.items() for candidate in values]
    return [RerankerRequest(target=word.lower(), candidates=_merge(ranked), threshold=0.0) for word, ranked in dataset.items()]

async def run_test(reranker: Reranker, requests: list[RerankerRequest], batch_size: int = 1):
    async def rerank(request: RerankerRequest, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await reranker(request)

    semaphore = asyncio.Semaphore(batch_size)
    tasks = [rerank(request, semaphore) for request in requests]
    results = await tqdm_asyncio.gather(*tasks, desc='Running tests')
    return results 
    


async def main():
    dataset_path = 'datasets/reranker.json'
    dataset = await prepare_dataset(dataset_path)
    '''batch_size = 10
    reranker = Reranker(oclient, model_name=os.environ['MODEL_NAME'], logger=logger)
    requests = prepare_requests(dataset)
    results = await run_test(reranker, requests, batch_size)
    await asave_json('test_results/reranker_results.json', results)'''
    results = await aread_json('test_results/reranker_results_0.json')
    results = results + await aread_json('test_results/reranker_results.json')
    metrics = evaluate(dataset, results)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv('test_results/reranker_metrics.tsv', sep='\t')
    print(metrics[['ndcg', 'tau', 'r']].mean())

if __name__ == "__main__":
    asyncio.run(main())