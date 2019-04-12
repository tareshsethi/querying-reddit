import ast
import pickle

save_prefix = 'data/'
save_filenames = ['total_preprocessed_small', 'subreddits_small', 'dataset_small']

def preprocess(filename):
    to_save = [[],[],[]] # list of preprocessed_raw_lines, list of subreddit titles, list of posts
    with open (filename) as f:
        counter = 0
        for line in f:
            line_ = ast.literal_eval(line)
            if '.jpg' in line_[2] or '.png' in line_[2]:
                to_save[0].append(line)
                to_save[1].append(line_[0])
                to_save[2].append(line_[1])
            if counter == 10000:
                break
            counter += 1

    for i, filename in enumerate(save_filenames):
        with open (save_prefix + filename + '.pkl', 'wb') as f:
            pickle.dump(to_save[i], f)

def load(filename='dataset'):
    with open (save_prefix + filename + '.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == '__main__':
    preprocess('safe_links_all')