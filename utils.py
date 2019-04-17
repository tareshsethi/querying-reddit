from gensim.scripts.glove2word2vec import glove2word2vec
import pickle
import torch.utils.data as data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, input_, output_):
        'Initialization'
        self.input_ = input_
        self.output_ = output_

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_)

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.input_[index]
        y = self.output_[index]
        return x, y

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