import json

from random import sample
from typing import List, Dict, Set, Optional
from collections import deque

from fire import Fire
from tqdm import tqdm

from taxoenrich.core import RuWordNet
from taxoenrich.data_utils import read_dataset
from utils.io_utils import save_json

# Initialize RuWordNet
wordnet = RuWordNet('wordnets/RuWordNet')
all_nodes = [sid for sid, synset in wordnet.synsets.items() if synset.synset_type == 'N']


def get_all_hypernyms(nodes: List[str], max_level: Optional[int] = None):
    dq = deque([(node, 0) for node in nodes])
    added = []
    while dq:
        node, level = dq.popleft()
        hypernyms = [_node['id'] for _node in wordnet.get_hypernyms(node, pos='N')]
        added.extend(hypernyms)
        if not max_level or max_level > level + 1:
            dq.extend([(_node, level + 1) for _node in hypernyms])
    return added

def get_all_hyponyms(nodes: List[str], max_level: Optional[int] = None):
    dq = deque([(node, 0) for node in nodes])
    added = []
    while dq:
        node, level = dq.popleft()
        hyponyms = [_node['id'] for _node in wordnet.get_hyponyms(node, pos='N')]
        added.extend(hyponyms)
        if not max_level or max_level > level + 1:
            dq.extend([(_node, level + 1) for _node in hyponyms])
    return added

def create_sample(true_hypernyms: List[str], max_nodes_cat: int = 5) -> Dict[int, Set]:
    ranked_nodes = {}
    known_synsets = set(true_hypernyms)

    # adding all grandparents
    grandparents = [wordnet.get_hypernyms(node_id, pos='N') for node_id in true_hypernyms]
    grandparents = [gp['id'] for nodes in grandparents for gp in nodes]
    ranked_nodes[5] = list(set(true_hypernyms + sample(grandparents, min(len(grandparents), max_nodes_cat))))
    known_synsets = known_synsets | set(grandparents)

    # adding rank-4 nodes: upper-level hypernyms
    ancestors = get_all_hypernyms(grandparents)
    ranked_nodes[4] = set(ancestors) - known_synsets
    known_synsets = known_synsets | ranked_nodes[4]

    # adding rank-3 nodes: cohyponyms
    parents = set(true_hypernyms) - set(grandparents)
    cohyponyms = get_all_hyponyms(parents, 1)
    ranked_nodes[3] = set(cohyponyms) - known_synsets
    known_synsets = known_synsets | ranked_nodes[3]

    # adding rank-2 nodes: distance > 2 descendants of grandparents 
    related = get_all_hyponyms(grandparents)
    ranked_nodes[2] = set(related) - known_synsets
    known_synsets = known_synsets | ranked_nodes[2]

    # adding rank-1 nodes: other random nodes
    ranked_nodes[1] = sample(list(set(all_nodes) - known_synsets), max_nodes_cat)

    # shorten samples except 5
    for i in range(2, 5):
        ranked_nodes[i] = sample(list(ranked_nodes[i]), min(len(ranked_nodes[i]), max_nodes_cat))

    return ranked_nodes

def main(dataset_path: str):
    # Load Dataset
    dataset = read_dataset(dataset_path, read_fn=json.loads)
    results = {word: create_sample(nodes[0], max_nodes_cat=3) for word, nodes in tqdm(dataset.items())}

    save_json('datasets/reranker.json', results)

if __name__ == '__main__':
    Fire(main)
