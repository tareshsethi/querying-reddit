from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from InferSent.encoder.models import InferSent

data_prefix = 'data/'
dataset_path = 'data/dataset.pkl'
subreddits_path = 'data/subreddits.pkl'
glove_to_word2vec_file = data_prefix + 'word2vec.txt'

DO_TFIDF = False

def fit_to_tfidf(data, do_tfidf=True):
    if do_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data)
        feature_names = vectorizer.get_feature_names()
        save (X, data_prefix + 'X.pkl')
        save (feature_names, data_prefix + 'feature_names.pkl')
    else:
        X = load (data_prefix + 'X.pkl')
        feature_names = load (data_prefix + 'feature_names.pkl')
    return X, feature_names

def train(dataset, do_tfidf=True):
    # Word2Vec
    word_embedding_model = KeyedVectors.load_word2vec_format(glove_to_word2vec_file, binary=False)
    word_vectors = word_embedding_model.wv

    print ('hey')

    # pass reddit posts through sentence encodings from facebook
    V = 2
    MODEL_PATH = 'InferSent/encoder/infersent%s.pickle' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    W2V_PATH = 'InferSent/fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab(dataset, tokenize=True)
    embeddings = infersent.encode(dataset, tokenize=True)


    print ('yay')

    return


    # fit to tfidf
    X, feature_names = fit_to_tfidf(dataset, do_tfidf=do_tfidf)

    return

    # initialize neural network
    features_in = list(word_vectors.values())[0]
    out = 100
    hidden_size = int((features_in + out)/ 2)
    model = nn.Sequential(
        nn.Linear(features_in, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out),
        nn.Sigmoid()
    )
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # training algorithm

    for epoch in range(50):
        y_pred = model(x)
        loss = F.cosine_similarity(y_pred, y)
        if epoch % 100:
            print('epoch: ', epoch,' loss: ', np.mean(loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    dataset = load(dataset_path)
    train(dataset, do_tfidf=DO_TFIDF)
