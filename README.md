# querying-reddit 

## Brief overview of files

* preprocess.py - Code for weeding out the examples that are not images (pngs or jpgs), and a place to add further preprocessing if necessary
* train.py - Code for training of second implementation below
* utils.py - toolbox including pickle and data conversions
* query.py - interface for getting the query, call after running train.py. Format - enter queries followed by weights as shown here
```bash
query -q apple banana -w 0.3 0.7
```

## Ideas

* The first possible implementation involves producing a tf-idf representation of the reddit posts, then maneuvering to gather the K keywords of all documents (words with highest sum/average tf-idf scores for each keyword over all documents). With these K keywords, generate a mapping from keywords to lists of the top k words where k < K, and where the lists are ranked in decreasing order of tf-idf scores for the corresponding keyword across documents. Then, train a Word2Vec model across the entire dataset of reddit posts and subreddit names, and when queried, simply find the embedding and use cosine similarity to find the most similar keyword and therefore the best resulting document (first element of the list that the keyword points to). Overall efficient training and inference as long as K is not too large, but the method lacks semantic meaning and relies much on bag of words to represent sentences.

* The implementation here involves a more semantic approach, passing a pre-trained Sentence Encoder (InferSent) through all subreddit posts to generate sentence-level embeddings. Universal Sentence Encoder could be a better replacement for the encoder, but requires tensorflow so it is not used here. Then, we employ a simple neural network that maps keyword embeddings from Word2Vec to sentence-level embeddings. We generate k keywords for each subreddit post using tf-idf similar to as described above, and take an average of these embeddings in the hope that we find a representative example query. Then, we train the neural network using a cosine-similarity metric between the generated sentence embedding and ground-truth embedding to track performance. Although we use pre-trained Word2Vec models and Universal Sentence Encoders in this implementation, it would be very useful to train to the dataset as reddit posts are much different than news/wikipedia/etc datasets (more slang for example). Multiple keywords and weights for queries would be easy here to support as we would just take a weighted average of the query keywords' embeddings using provided weights and pass the generated embedding through the neural network. Although we now have semantic meaning, one major pitfall is that the output of the neural network is continuous and large and therefore may result in degenerate cases/inefficient learning. One possible way to avoid this could be employing a clustering technique for the subreddit posts and having the neural network produce a class indicating the cluster with a cross entropy loss, and then use some other ranking function to rank the subreddit posts within the cluster.

* Approaches drawn from probability like mutual information between query and sentence-level embeddings could also be quite interesting to try, example paper found here - https://arxiv.org/abs/1807.06653. One thing to continue thinking about would be how to utilize subreddit names - would they be keywords?
