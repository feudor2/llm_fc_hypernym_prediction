import asyncio
import re

import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import product
from copy import deepcopy
from typing import Any

from taxoenrich.core import RuWordNet
from taxoenrich.utils import read_dataset, compute_ap, compute_rr
from utils.data_processing import collect_tracking_results, load_start_nodes
from utils.io_utils import save_json 


wordnet = RuWordNet('././wordnets/RuWordNet')

def compute_p(true, pred, k=1):
    k = min(k, len(true))
    top_k_pred = pred[:k]
    num_relevant_in_top_k = np.sum(np.isin(top_k_pred, true))
    return num_relevant_in_top_k / k

def merge_words(results: list[dict[str, Any]]):
    output = defaultdict(list)
    for res in results:
        output[res['word'].upper()].append(res['synsets'][::-1])
    return output

def select_candidates(candidates: dict[str, list[list[str]]], longest_first: bool = True, deepest_first: bool = False, selected_paths: list[int] = [], k: int = 15):
    """
    Select top k candidates from selected_paths by strategy:
    - longest_first (longest_first=True, deepest_first=False): from last to first path by path, paths ordered by length in descending order
    - shortest_first (longest_first=False, deepest_first=False): from last to first path by path, paths ordered by length in ascending order
    - deepest_first (deepest_first=True): from deepest to first by highest (by topological generation), paths ordered by length depending on longest_first parameter
    """
    # Selecting paths
    #print(f'Started:  longest_first={longest_first}, deepest_first={deepest_first}, selected_paths={selected_paths}')
    selected_candidates = []
    if selected_paths:
        for word, res in deepcopy(candidates).items():
            selected_results = [res[i] for i in selected_paths if i < len(res)]
            selected_candidates.append((word, selected_results))
    else:
        selected_candidates = list(deepcopy(candidates).items())
    # Sorting candidates
    _print = False
    ranked_items = defaultdict(list)
    for word, res in selected_candidates:
        sorted_res = sorted(res, key=len, reverse=longest_first)
        if sorted_res:
            if _print:
                print(f'Sorted result={sorted_res}')
            if not deepest_first:
                for i in range(len(sorted_res[int(longest_first)-1])):
                    for j in range(len(sorted_res)):
                        if _print:
                            print(f'Current iteration: i={i}, j={j}, sorted_res[j]={sorted_res[j]}, ranked_items[word]={ranked_items[word]}')
                        if sorted_res[j]:
                            current_node = sorted_res[j].pop(-1)
                            if current_node not in ranked_items[word]:
                                ranked_items[word].append(current_node)
            else:
                merged_res = list(set([node for _res in sorted_res for node in _res]))
                if _print:
                    print(f'merged_res={merged_res}')
                ranked_items[word] = sorted(merged_res, key=lambda x: wordnet.find_generation(x), reverse=True)

            ranked_items[word] = ranked_items[word][:k]
        else:
            ranked_items[word] = []

    return ranked_items
    

async def main(tracking_path: str, start_node: bool):
    dataset_path = '././test_results/context_analyser_results.tsv'
    dataset = read_dataset(dataset_path, read_fn=lambda x: eval(x))
    #dataset = load_dataset(dataset_path)
    print('Loaded dataset:', list(dataset.items())[0], ', ...')
    results = await collect_tracking_results(tracking_path, start_node)
    results = merge_words(results, start_node)
    print('Loaded results:', *list(results.items())[:1], ', ...')
    selected_words = [word for word, values in results.items() if any(values) and word in dataset]
    #print('Selected words:', selected_words[0], ', ...')
    yandex_results = load_start_nodes('././examples/fasttext_baseline.json')
    print('External results:', list(yandex_results.items())[:5])
    candidates = {
        #'longest_first': select_candidates(results, longest_first=True, deepest_first=False),
        #'shortest_first': select_candidates(results, longest_first=False, deepest_first=False),
        #'deepest_first': select_candidates(results, longest_first=True, deepest_first=True),
        'TaxoExplore': results,
        'fine-tuning (baseline)': yandex_results
    } 
    '''| {
        f'longest_first [{str(i + 1)}]': select_candidates(results, longest_first=True, deepest_first=False, selected_paths=[i]) for i in range(0, 3)
    } | {
        f'shortest_first [{str(i + 1)}]': select_candidates(results, longest_first=False, deepest_first=False, selected_paths=[i]) for i in range(0, 3)
    } | {
        f'deepest_first [{str(i + 1)}]': select_candidates(results, longest_first=True, deepest_first=True, selected_paths=[i]) for i in range(0, 3)
    }'''

    for method, _candidates in candidates.items():
        save_json(f'././test_results/candidates/{method}.json', _candidates)
    
    metrics = []
    for method, _candidates in candidates.items():
        for word, node_ids in _candidates.items():
            if word in selected_words:
                '''ap = compute_ap(dataset[word], node_ids, k=15)
                rr = compute_rr([node for nodes in dataset[word] for node in nodes], node_ids, k=15)'''
                if 'baseline' not in method:
                    for pred_ids, true_ids in zip(node_ids, dataset[word]):
                        p1 = compute_p(true_ids, pred_ids)
                        ap1 = compute_ap([true_ids], pred_ids, k=1 if start_node else 15)
                        rr1 = compute_rr(true_ids, pred_ids, k=1 if start_node else 15)
                        p3 = compute_p(true_ids, pred_ids, k=3)
                        ap3 = compute_ap([true_ids], pred_ids, k=3 if start_node else 15)
                        rr3 = compute_rr(true_ids, pred_ids, k=3 if start_node else 15)
                else:
                    for true_ids in dataset[word]:
                        p3 = compute_p(true_ids, node_ids, k=3)
                        ap3 = compute_ap([true_ids], node_ids, k=3 if start_node else 15)
                        rr3 = compute_rr(true_ids, node_ids, k=3 if start_node else 15)
                        p1 = compute_p(true_ids, node_ids, k=1)
                        ap1 = compute_ap([true_ids], node_ids, k=1 if start_node else 15)
                        rr1 = compute_rr(true_ids, node_ids, k=1 if start_node else 15)
                metrics.append({
                    'method': method,
                    'word': word,
                    'pred': node_ids,
                    'true': dataset[word],
                    'MRR@1': rr1,
                    'MAP@1': ap1,
                    'P@1': p1,
                    'MRR@3': rr3,
                    'MAP@3': ap3,
                    'P@3': p3
                })
                    
    
    df = pd.DataFrame(metrics)

    '''metrics = {
        method: get_scores(_candidates, dataset, k=15, selected=selected_words) for method, _candidates in candidates.items()
    }

    df = pd.DataFrame([{'method': method, 'MAP': scores[0], 'MRR': scores[1]} for method, scores in metrics.items()])
    grouped_methods = pd.DataFrame({'group': df['method'].apply(lambda m: re.sub( '\[\d\]', '1 (mean)', m))})
    grouped = pd.concat([df, grouped_methods], axis=1)
    group = grouped[grouped['group'].str.contains('(mean)')].groupby('group')[['MAP', 'MRR']].mean()
    df = pd.concat([df, group.reset_index().rename(columns={'group': 'method'})]) '''
    df.to_csv(f'././test_results/metrics/{tracking_path.split("/")[1]}_items.tsv', sep='\t', index=False)

def process_df(tracking_path):
    df = pd.read_csv(f'././test_results/metrics/{tracking_path.split("/")[1]}_items.tsv', sep='\t')
    df = df.drop(columns=['word', 'pred', 'true']).groupby('method').mean().reset_index()
    grouped_methods = pd.DataFrame({'group': df['method'].apply(lambda m: re.sub( '\[\d\]', '1 (mean)', m))})
    grouped = pd.concat([df, grouped_methods], axis=1)
    group = grouped[grouped['group'].str.contains('(mean)', regex=False)].groupby('group')[[_metric + f'@{k}' for _metric, k in product(['MAP', 'MRR', 'P'], [1, 3])]].mean()
    df = pd.concat([df, group.reset_index().rename(columns={'group': 'method'})]).round(3)
    df.to_csv(f'././test_results/metrics/{tracking_path.split("/")[1]}_metrics.tsv', sep='\t', index=False)


if __name__ == "__main__":
    #tracking_path = '././tracking_results/batch_1768282585'
    tracking_path = '././tracking_results/batch_1768400615'
    asyncio.run(main(tracking_path, start_node=True))
    process_df(tracking_path)