from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import numpy as np

data_prefix = 'data/'
dataset_path = 'data/dataset.pkl'
subreddits_path = 'data/subreddits.pkl'
glove_to_word2vec_file = data_prefix + 'word2vec.txt'

DO_TFIDF = True

def fit_to_tfidf(data, to_save=True):
    if to_save:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data)
        feature_names = vectorizer.get_feature_names()
        save (X, data_prefix + 'X.pkl')
        save (feature_names, data_prefix + 'feature_names.pkl')
    else:
        print ('yay')
        X = load (data_prefix + 'X.pkl')
        feature_names = load (data_prefix + 'feature_names.pkl')
    return X, feature_names

def train(dataset, do_tfidf=True):
    word_embedding_model = KeyedVectors.load_word2vec_format(glove_to_word2vec_file, binary=False)
    X, feature_names = fit_to_tfidf(dataset, to_save=False)
    print (np.squeeze(X[0].toarray()))

if __name__ == '__main__':
    dataset = load(dataset_path)
    train(dataset, do_tfidf=DO_TFIDF)

