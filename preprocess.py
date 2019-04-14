import ast
import pickle
from joblib import Parallel, delayed
import os

save_prefix = 'data/'
splitted_data_prefix = save_prefix + 'splitted/'
save_filenames = ['total_preprocessed', 'subreddits', 'dataset']
data_checkpoints_prefix = save_prefix + 'data_checkpoints/'
SAVE_CHECKPOINTS = True

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

    if SAVE_CHECKPOINTS:
        for i, file_ in enumerate(save_filenames):
            with open (data_checkpoints_prefix + file_ + '_' + filename.split(os.sep)[-1] + '.pkl', 'wb') as f:
                pickle.dump(to_save[i], f)

    print ('finish job')

    return to_save

def preprocess(directory_of_splitted_files):
    for r, d, f in os.walk(directory_of_splitted_files):
        arg_instances = [splitted_data_prefix + s for s in f]
        break

    print (arg_instances)
    
    to_save = Parallel(n_jobs=-1, verbose=5, backend="threading")(
        map(delayed(preprocess_job), arg_instances))

    for i, filename in enumerate(save_filenames):
        with open (save_prefix + filename + '.pkl', 'wb') as f:
            pickle.dump(to_save[i], f)

def load(filename='dataset'):
    with open (save_prefix + filename + '.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == '__main__':
    preprocess(save_prefix + 'splitted/')