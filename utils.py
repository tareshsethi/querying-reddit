from gensim.scripts.glove2word2vec import glove2word2vec
import pickle

def convert_glove_to_word2vec():
    glove_input_file = data_prefix + 'glove.6B.100d.txt'
    word2vec_output_file = data_prefix + 'word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)

def load(filepath):
    with open (filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def save(data, filepath):
    with open (filepath, 'wb') as f:
        pickle.dump(data, f)