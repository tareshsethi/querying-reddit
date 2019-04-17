import torch
from gensim.models import KeyedVectors
from utils import *
import torch.nn.functional as F
import argparse

data_prefix = 'data/'
glove_to_word2vec_file = data_prefix + 'word2vec.txt'
dataset_path = 'data/dataset.pkl'

def query(queries, weights, model, word_vectors, embeddings, dataset):
    output = model(sum([word_vectors[queries[i]] * weights[i] for i in len(queries)]))
    outputs = np.array([output for _ in embeddings.shape[0]])
    doc_index = np.argmax(F.cosine_similarity(outputs, embeddings))
    doc = dataset[doc_index]
    print ('Closest match:')
    print (doc)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Retrieve best doc given queries')
    parser.add_argument('-q','--queries', metavar='queries', nargs='+', help='Enter queries', required=True)
    parser.add_argument('-w','--weights', metavar='weights', nargs='+', help='Enter queries weights', required=True)
    args = parser.parse_args()
    queries = args.queries
    weights = [float(s) for s in args.weights]

    # load neural network model
    model = torch.load(data_prefix + str(epoch) + '_model.pt')
    
    # load word embedding model
    word_embedding_model = KeyedVectors.load_word2vec_format(glove_to_word2vec_file, binary=False)
    word_vectors = word_embedding_model.wv

    # load sentence embeddings
    embeddings = load(data_prefix + 'embeddings.pkl')

    # load dataset of sentences
    dataset = load(dataset_path)

    # run the query
    query(queries, weights, model, word_vectors, embeddings, dataset)