from preprocess import load

PREPROCESS = False
dataset_filename = 'dataset'

def main ():
    dataset = load(dataset_filename)
    print (len(dataset))
    print ('yay')
    # fit_to_tfidf(['hi how how how hello', 'how how how are you', 'hows how everything', 'whats up dawg'])
    # print ('finished')

if __name__ == '__main__':
    main ()