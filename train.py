from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from InferSent.encoder.models import InferSent
import torch.utils.data as data
from joblib import Parallel, delayed
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from torch.autograd import Variable

data_prefix = '/media/sdc1/extra_space/data/'
dataset_path = '/media/sdc1/extra_space/data/dataset.pkl'
subreddits_path = '/media/sdc1/extra_space/data/subreddits.pkl'
glove_to_word2vec_file = data_prefix + 'word2vec.txt'

DO_TFIDF = False
USE_SMALL_DATSET = True
GENERATE_SENTENCE_EMBEDDINGS = False
GENERATE_KEYWORDS_EMBEDDINGS_LIST_MATRIX = False

def fit_to_tfidf(data, do_tfidf=True):
    if do_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data)
        keywords_per_document = vectorizer.inverse_transform(X)
        feature_names = vectorizer.get_feature_names()
        save (X, data_prefix + 'X.pkl')
        save (keywords_per_document, data_prefix + 'keywords_per_document.pkl')
        save (feature_names, data_prefix + 'feature_names.pkl')
    else:
        X = load (data_prefix + 'X.pkl')
        keywords_per_document = load (data_prefix + 'keywords_per_document.pkl')
        feature_names = load (data_prefix + 'feature_names.pkl')
    return X, keywords_per_document, feature_names

def get_keyword_embeddings_list_matrix(X, keywords_per_document, word_vectors, num_sentences, word_embedding_size, top_n=5):
    matrix = np.zeros((num_sentences, word_embedding_size))
    for i in range (num_sentences):
        count = 0
        for feat in keywords_per_document[i]:
            if feat in word_vectors:
                matrix[i] += np.array(word_vectors[feat])
                count += 1
        if count != 0:
            matrix[i] /= count
    return matrix

def train(dataset, do_tfidf=True, use_small_dataset=True, generate_sentence_embeddings=True, generate_keyword_embeddings_list_matrix=True):
    if use_small_dataset:
        dataset = dataset[:20000]
    num_sentences = len(dataset)
    print (num_sentences)

    # Word2Vec
    word_embedding_model = KeyedVectors.load_word2vec_format(glove_to_word2vec_file, binary=False)
    word_vectors = word_embedding_model.wv
    word_embedding_size = word_vectors['the'].shape[0]

    if generate_sentence_embeddings:
        # pass reddit posts through sentence encodings from facebook
        V = 2
        MODEL_PATH = 'InferSent/encoder/infersent%s.pickle' % V
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = data_prefix + 'crawl-300d-2M-subword.vec'
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab(dataset, tokenize=True)
        embeddings = infersent.encode(dataset, tokenize=True)
        save(embeddings, data_prefix + 'embeddings.pkl')
    else:
        embeddings = load(data_prefix + 'embeddings.pkl')

    # fit to tfidf
    X, keywords_per_document, feature_names = fit_to_tfidf(dataset, do_tfidf=do_tfidf)
    if generate_keyword_embeddings_list_matrix:
        keyword_embeddings_list_matrix = get_keyword_embeddings_list_matrix(X, keywords_per_document, word_vectors, num_sentences, word_embedding_size, top_n=5)
        save(keyword_embeddings_list_matrix, data_prefix + 'matrix_keyword_embeddings.pkl')
    else:
        keyword_embeddings_list_matrix = load(data_prefix + 'matrix_keyword_embeddings.pkl')

    # initialize neural network
    features_in = word_embedding_size
    out = 4096
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
    
    # set up datset
    dataset = Dataset(keyword_embeddings_list_matrix, embeddings)
    datasets = {}
    indices = list(range(num_sentences))
    np.random.shuffle(indices)
    split = int(np.floor(0.8 * num_sentences))
    train_indices, val_indices = indices[:split], indices[split:]
    datasets['train'], datasets['val'] = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
    dataloaders = {x: torch.utils.data.DataLoader(dataset, sampler=datasets[x], batch_size=4,
        num_workers=4) for x in ['train', 'val']
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    # training algorithm - https://jhui.github.io/2018/02/09/PyTorch-neural-networks/
    num_epochs = 50
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            best_score = 0.0

            for data in dataloaders[phase]:
                inputs, ground_truth_outputs = data
                inputs, ground_truth_outputs = Variable(inputs), Variable(ground_truth_outputs)

                optimizer.zero_grad()
                outputs = model(inputs.float())

                loss = -1 * torch.mean(F.cosine_similarity(outputs, ground_truth_outputs))
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss * inputs.size(0)
                running_score = -1 * running_loss
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_score = running_score / dataset_sizes[phase]
            
            if phase == 'val' and epoch_score > best_score:
                best_score = epoch_score
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print ('epoch ' + str(epoch) + ' epoch_loss: ' + str(epoch_loss))
        
        model.load_state_dict(best_model_wts)
        torch.save(model, data_prefix + str(epoch) + '_model.pt')

if __name__ == '__main__':
    dataset = load(dataset_path)
    train(dataset, do_tfidf=DO_TFIDF, use_small_dataset=USE_SMALL_DATSET,
        generate_sentence_embeddings=GENERATE_SENTENCE_EMBEDDINGS, 
        generate_keyword_embeddings_list_matrix=GENERATE_KEYWORDS_EMBEDDINGS_LIST_MATRIX
    )
