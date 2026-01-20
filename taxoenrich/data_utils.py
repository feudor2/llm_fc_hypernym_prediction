import codecs
from collections import defaultdict

def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0].upper()
            hypernyms = read_fn(line_split[1])
            vocab[word].append(hypernyms)
    return vocab