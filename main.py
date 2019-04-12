from preprocess import load

PREPROCESS = False
dataset_filename = 'dataset_small'

def main ():
    if PREPROCESS:
        preprocess ('safe_links_all')
    dataset = load(dataset_filename)
    print (len(dataset))
    print ('yay')
    # fit_to_tfidf(['hi how how how hello', 'how how how are you', 'hows how everything', 'whats up dawg'])
    # print ('finished')

if __name__ == '__main__':
    main ()