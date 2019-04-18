import torch
from gensim.models import KeyedVectors
from utils import *
import torch.nn.functional as F
import argparse
import numpy as np

data_prefix = '/media/sdc1/extra_space/data/'
glove_to_word2vec_file = data_prefix + 'word2vec.txt'
dataset_path = data_prefix + 'dataset.pkl'
USE_SMALL_DATASET = True

def query(queries, weights, model, word_vectors, embeddings, dataset):
    input_ = torch.cat([torch.from_numpy(word_vectors[queries[i]] * weights[i]) for i in range(len(queries))])
    if len(queries) > 1:
        input_ = torch.sum(input_, 1)
    output = model(input_)
    outputs = torch.cat([output.unsqueeze(0) for i in range(len(dataset))], 0)
    outputs = torch.zeros(20000,4096)
    top_5_doc_indices_sorted = np.argsort(-1 * F.cosine_similarity(outputs, torch.from_numpy(embeddings)),0)[:5]
    docs = [dataset[doc_index] for doc_index in top_5_doc_indices_sorted]
    print ('Closest 5 matches:')
    print (docs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Retrieve best doc given queries')
    parser.add_argument('-q','--queries', metavar='queries', nargs='+', help='Enter queries', required=True)
    parser.add_argument('-w','--weights', metavar='weights', nargs='+', help='Enter queries weights', required=True)
    args = parser.parse_args()
    queries = args.queries
    weights = [float(s) for s in args.weights]

    # load neural network model
    model = torch.load(data_prefix + '40_model.pt')
    
    # load word embedding model
    word_embedding_model = KeyedVectors.load_word2vec_format(glove_to_word2vec_file, binary=False)
    word_vectors = word_embedding_model.wv

    # load sentence embeddings
    embeddings = load(data_prefix + 'embeddings.pkl')

    # load dataset of sentences
    dataset = load(dataset_path)
    if USE_SMALL_DATASET:
        dataset = dataset[:20000]

    # check input
    for query_ in queries:
        assert query_ in word_vectors
    assert sum(weights) == 1.0

    # run the query
    query(queries, weights, model, word_vectors, embeddings, dataset)