import ast
import pickle
from joblib import Parallel, delayed
import os

save_prefix = 'data/'
splitted_data_prefix = save_prefix + 'splitted/'
save_filenames = ['total_preprocessed', 'subreddits', 'dataset']
data_checkpoints_prefix = save_prefix + 'data_checkpoints/'
SAVE_CHECKPOINTS = False
FROM_CHECKPOINTS = True

def preprocess_job(filename):

    print ('start job')

    to_save = [[],[],[]] # list of preprocessed_raw_lines, list of subreddit titles, list of posts
    with open (filename) as f:
        for line in f:
            try:
                line_ = ast.literal_eval(line)
                if '.jpg' in line_[2] or '.png' in line_[2]:
                    to_save[0].append(line)
                    to_save[1].append(line_[0])
                    to_save[2].append(line_[1])
            except ValueError:
                pass  # do nothing!

    # save batches of preprocessed data
    if SAVE_CHECKPOINTS:
        for i, file_ in enumerate(save_filenames):
            with open (data_checkpoints_prefix + file_ + '_' + filename.split(os.sep)[-1] + '.pkl', 'wb') as f:
                pickle.dump(to_save[i], f)

    print ('finish job')

    return to_save

def preprocess(directory, from_checkpoints=False):
    # conglomerate previously preprocessed batches into one and save
    if from_checkpoints:
        to_save = [[],[],[]]

        for i, filename in enumerate(save_filenames):
            for r, d, f in os.walk(directory):
                individuals = [directory + s for s in f if filename in s]
                break

            for file_ in individuals: 
                with open (file_, 'rb') as f:
                    to_save[i] = to_save[i] + pickle.load(f)

    # preprocess by batches using multithreading then save
    else:
        for r, d, f in os.walk(directory):
            arg_instances = [directory + s for s in f]
            break
        
        to_save = Parallel(n_jobs=-1, verbose=5, backend="threading")(
            map(delayed(preprocess_job), arg_instances))

    # save entire dataset via pickle   
    for i, filename in enumerate(save_filenames):
        with open (save_prefix + filename + '.pkl', 'wb') as f:
            pickle.dump(to_save[i], f)

def load(filename='dataset'):
    with open (save_prefix + filename + '.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == '__main__':
    if FROM_CHECKPOINTS:
        preprocess(data_checkpoints_prefix, from_checkpoints=FROM_CHECKPOINTS)
    else:
        preprocess(splitted_data_prefix, from_checkpoints=FROM_CHECKPOINTS)
