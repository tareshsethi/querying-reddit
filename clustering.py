from sklearn.feature_extraction.text import TfidfVectorizer

def fit_to_tfidf(data):
    # vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    print(vectorizer.get_feature_names())
    print (X)
    print (type(X))

fit_to_tfidf(['hi how how how hello', 'how how how are you', 'hows how everything', 'whats up dawg'])
