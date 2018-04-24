
import os
import glob
import pickle
import string
from sklearn.utils import shuffle
 # Nice to time some stuff
from datetime import datetime
# BeautifulSoup to easily remove HTML tags
from bs4 import BeautifulSoup 
# RegEx for removing non-letter characters
import re
#Natural Language ToolKit, for importing prdefined Stopwords and word stemmer 
import nltk
# import stopwords, see http://www.nltk.org/nltk_data/ for more interesting data.
from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays
from sklearn.externals import joblib

cache_dir = ""   # Directory where to keep cached files

def run_sentimentLab1():

    print("Hello Sentiment Analysis Lab")

    print("Loading IMDB reviews data...")
    start_time = datetime.now()
    data, labels = read_imdb_data()
    print("Done Loading!")
    print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time ))
    print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format( len(data['train']['pos']), len(data['train']['neg']), len(data['test']['pos']), len(data['test']['neg'])))

    
    print("\n************************************************")
    print("Showing example of positive review")
    print(data['train']['pos'][2])

    print("\n************************************************")
    print("Showing example of negative review")
    print(data['train']['neg'][2])

    # download list of stopwords (only once; need not run it again), usually saved in NLTK module folder
    print("\nChecking if stopwords list exists...")
    nltk.download("stopwords")   
    

    data_train, data_test, labels_train, labels_test = prepare_imdb_data(data)
    print("\n\nIMDb reviews (combined): train = {}, test = {}".format(len(data_train), len(data_test)))
   
    #testing to remove non-standard characters and more
    testString =  """<HTML><HEAD>hello head</HEAD> <SCRIPT>and remove me too</SCRIPT> <BODY> Yees! AI is cool stuff! ÅÄÖäåö</BODY></HTML>"""
    print ("\nUsing Porterstemmer and stopwordslist to test on a HTML example")
    print("String before stemming and removal of stopwords\n" + testString)
    print("\nString after:\n")
    print(review_to_words(testString))
    print()


    cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
    os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists
     
    print("\nPreprocessing data...")
    print("Cache directory is " + cache_dir)

    words_train, words_test, labels_train, labels_test = preprocess_data(data_train, data_test, labels_train, labels_test, cache_dir)

    # Take a look at a sample
    print("\n--- Raw review ---")                                                                                               
    print(data_train[1])
    print("\n--- Preprocessed words ---")
    print(words_train[1])
    print("\n--- Label ---")
    print(labels_train[1])

    # Extract Bag of Words features for both training and test datasets
    features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test, cache_dir=cache_dir)

    # Inspect the vocabulary that was computed
    print("Vocabulary: {} words".format(len(vocabulary)))

    import random
    print("Sample words: {}".format(random.sample(list(vocabulary.keys()), 8)))

    # Sample
    print("\n--- Preprocessed words ---")
    print(words_train[5])
    print("\n--- Bag-of-Words features ---")
    print(features_train[5])
    print("\n--- Label ---")
    print(labels_train[5])

    # Plot the BoW feature vector for a training document
    plt.plot(features_train[5,:])
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title("Bag of Words (BoW) Feature Vector")
    plt.show()

    print("Sparsity level = {0:.1f}%".format(100 * np.count_nonzero(features_train==0)/features_train.size))


    # Find number of occurrences for each word in the training set
    word_freq = features_train.sum(axis=0)

    # Sort it in descending order
    sorted_word_freq = np.sort(word_freq)[::-1]

    # Plot 
    plt.plot(sorted_word_freq)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Number of occurrences')
    plt.title("Number of occurrences for each word in the training set")
    plt.show()

    print("First most frequent word occurrences: " + str(sorted_word_freq[0]))
    print("Second most frequent word occurrences: " + str(sorted_word_freq[1]))


    #Normalize BoW features in training and test set
    import sklearn.preprocessing as pr
    # Does "normalize" here mean minmax scaling or true normalization?  Better results with normalize() so use that...
    features_train = pr.normalize(features_train)
    features_test  = pr.normalize(features_test)

    from sklearn.naive_bayes import GaussianNB

    # Train a Guassian Naive Bayes classifier
    clf1 = GaussianNB()
    clf1.fit(features_train, labels_train)


    # Now that the data has all been properly transformed, we can feed it into a classifier. To get a baseline model, we train a Naive Bayes classifier 
    # from scikit-learn (specifically, GaussianNB), and evaluate its accuracy on the test set.
    # Calculate the mean accuracy score on training and test sets
    print("[{}] Accuracy: train = {}, test = {}".format(clf1.__class__.__name__, clf1.score(features_train, labels_train), clf1.score(features_test, labels_test)))

    """
    Use GradientBoostingClassifier from scikit-learn to classify the BoW data. This model has a number of parameters.
    We use default parameters for some of them and pre-set the rest for you, except one: n_estimators. 
    Find a proper value for this hyperparameter, use it to classify the data, and report how much improvement you get over Naive Bayes in terms of accuracy.

    """
    # See classify_gboost() method below for details
    n_estimators = 50
    clf2 = classify_gboost(features_train, features_test, labels_train, labels_test)

    n_estimators = 200
    clf3 = classify_gboost(features_train, features_test, labels_train, labels_test)

    n_estimators = 500
    clf4 = classify_gboost(features_train, features_test, labels_train, labels_test)


    from sklearn.feature_extraction.text import CountVectorizer
    # Write a sample review and set its true sentiment

    #Include irony and negated words. 
    my_review = "The movie was NOT a disappointment, it had loads of FX stuff. Almost as good as Aliens 3. I would not recommend this type of move to a kid, but surely to my friends"
    true_sentiment = 'pos'  # sentiment must be 'pos' or 'neg'

    #Apply the same preprocessing and vectorizing steps as you did for your training data

    #Do some stowording and stemming. :-)
    my_review_words = review_to_words(my_review)


    #cut and pasted from above, with some modifications
    countVectorizer = CountVectorizer(vocabulary=vocabulary, preprocessor=lambda x: x, tokenizer=lambda x: x)

    features_review = countVectorizer.fit_transform(my_review_words).toarray()


    # Then call your classifier to label it
    print("Prediction 1:")

    clf1.predict(features_review)[0]

    print("Prediction 2:")
    clf2.predict(features_review)[0]

    print("Prediction 3:")
    clf3.predict(features_review)[0]

    print("Prediction 4:")
    clf4.predict(features_review)[0]


    # We just saw how the task of sentiment analysis can be solved via a traditional machine learning approach: BoW + a nonlinear classifier. 
    # We now switch gears and use Recurrent Neural Networks, and in particular LSTMs, to perform sentiment analysis in Keras. 
    # Conveniently, Keras has a built-in IMDb movie reviews dataset that we can use, with the same vocabulary size.
    

    from keras.datasets import imdb  # import the built-in imdb dataset in Keras

    # Set the vocabulary size
    vocabulary_size = 5000

    # Load in training and test data (note the difference in convention compared to scikit-learn)
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    print("Loaded dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))
    # Inspect a sample review and its label
    print("--- Review ---")
    print(X_train[7])
    print("--- Label ---")
    print(y_train[7])


    # Notice that the label is an integer (0 for negative, 1 for positive), and the review itself is stored as a sequence of integers. 
    # These are word IDs that have been preassigned to individual words. 
    # To map them back to the original words, you can use the dictionary returned by imdb.get_word_index().

    # Map word IDs back to words
    word2id = imdb.get_word_index()
    id2word = {i: word for word, i in word2id.items()}
    print("--- Review (with words) ---")
    print([id2word.get(i, " ") for i in X_train[7]])
    print("--- Label ---")
    print(y_train[7])

    # The maximum and minimum review length (in terms of number of words) in the training set 

    #Loop through and then ask the new array for answers
    all_review_lengths = [len(x) for x in X_train]
    print("Total Reviews:" + str(len(all_review_lengths)))
    print("Max review length:"+ str(max(all_review_lengths)))
    print("Min review length:"+ str(min(all_review_lengths)))


    print("Done!")
    
    

                        
################################################################
# Read IMDb movie reviews from given directory.
################################################################
def read_imdb_data(data_dir='data/imdb-reviews'):
    """
    Directory structure expected:
    - data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/
    
    """

    #Data is from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    #Please read http://ai.stanford.edu/~amaas/data/sentiment/ for details. 
    #TODO: add automagic download of data if it does not exist.


    # Data, labels to be returned in nested dicts matching the dir. structure
    data = {}
    labels = {}

    # Assume 2 sub-directories: train, test
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        # Assume 2 sub-directories for sentiment (label): pos, neg
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            # Fetch list of files for this sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            # Read reviews data and assign labels
            for f in files:
                
                # with open(f) as review:  
                with open(f, encoding="utf8") as review: #Added UTF-8 
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)
            
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), "{}/{} data size does not match labels size".format(data_type, sentiment)
    
    # Return data, labels as nested dicts
    return data, labels


################################################################
# Prepare training and test sets from IMDb movie reviews.
################################################################
def prepare_imdb_data(data):
    
    # Combine positive and negative reviews and labels
    data_test = data['test']['pos'] + data['test']['neg']
    labels_test = ['pos'] * len(data['test']['pos']) +  ['neg'] * len(data['test']['neg'])
    
    data_train= data['train']['pos'] + data['train']['neg']
    labels_train= ['pos'] * len(data['train']['pos']) +  ['neg'] * len(data['train']['neg'])
    
    # Shuffle reviews and corresponding labels within training and test sets
    data_test = shuffle(data_test)
    labels_test = shuffle(labels_test)
    data_train= shuffle(data_train)
    labels_train= shuffle(labels_train)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


################################################################
# Convert a raw review string into a sequence of words.
################################################################
def review_to_words(review):

    #Remove HTML tags and non-letters using Beautifulsoup
    #Keep only ascii stuff to start with 
    
    soup = BeautifulSoup(review, "html5lib")
    soup = soup.extract()

    #From NLTK
    stemmer = PorterStemmer()
       
    #print("\nconvert to lowercase, tokenize, remove stopwords and stem")

    cleanerText = soup.get_text().lower()
    #cleanerText = ''.join(e for e in cleanerText if e.isalnum())
    cleanerText = re.sub('[^A-Za-z0-9 ]+', '', cleanerText)
    #print(cleanerText)
    
    #Todo: also remove <script> and <style> tags in next version
    tokenizedText = nltk.word_tokenize(cleanerText)
    #print(tokenizedText)
    
    nonstop_words = [word for word in tokenizedText if word not in stopwords.words('english')]
    words = [stemmer.stem(word) for word in nonstop_words]
    
    #print("Stemmed version:" + ' '.join(words))

    # Return final list of words
    return words


################################################################
# Convert each review to words; read from cache if available.
################################################################
def preprocess_data(data_train, data_test, labels_train, labels_test, cache_dir, cache_file="preprocessed_data.pkl"):


    #If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
                print("Read preprocessed data from cache file:", cache_file)
        except:
            print("No cached data found")
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        print("As no cache exists, Preprocess training and test data to obtain words for each review and add to cache. \nThis might take a few hours...")
        start_time = datetime.now()

        # Lets compare multithreaded map methods vs single thread
        # if(1):
           #beer =  with Pool(8) as p: p.(map(review_to_words, data_train)
           #beer2 = with Pool(8) as p: p.(map(review_to_words, data_test)
       
        """
        from multiprocessing.dummy import Pool as ThreadPool 
        pool = ThreadPool(4) 
        words_train = list(pool.map(review_to_words, data_train))
        words_test = list(pool.map(review_to_words, data_test))
        """
         
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))
        
        print("Done Preprocessing!")
        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time ))
        
        # Write to cache file for future runs
        if cache_file is not None:
            print("Retrieving data from cache file")
            cache_data = dict(words_train=words_train, words_test=words_test, labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
                print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        print("Unpack data loaded from cache file")
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'], cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test


 
################################################################
# Extract Bag-of-Words for a given set of documents, already preprocessed into words.
################################################################
def extract_BoW_features(words_train, words_test, vocabulary_size=5000, cache_dir=cache_dir, cache_file="bow_features.pkl"):
    
    print("Words train size:" + str(len(words_train)))
    print("Words test size:" + str(len(words_test)))
    print("Will check cache directory for existing data:" + str(cache_dir))
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(max_features=vocabulary_size, preprocessor=lambda x: x, tokenizer=lambda x: x)
        features_train = vectorizer.fit_transform(words_train).toarray()

        # Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.fit_transform(words_test).toarray()
        
        # NOTE: Remember to convert the features using .toarray() for a compact representation
        
        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test, vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'], cache_data['features_test'], cache_data['vocabulary'])
        print("Vocabulary size from cache:" + str(len(vocabulary)))
    
    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary





def classify_gboost(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV        
    # Initialize classifier
    clf = GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)

    # Classify the data using GradientBoostingClassifier
    # TODO(optional): Perform hyperparameter tuning / model selection
    gsearch = GridSearchCV(estimator = clf, param_grid = {'n_estimators': range(20, 81, 10)}, verbose=50, n_jobs=1)      #n_jobs must be 1 for windows, so no parallisation for now.
    gsearch.fit(X_train, y_train)
    best_clf = gsearch.best_estimator_
    print(gsearch.best_params_)
    
    # Print final training & test accuracy
    acc = best_clf.score(X_test, y_test)
    print("[{}] Accuracy: train = {}, test = {}".format(
        best_clf.__class__.__name__,
        best_clf.score(X_train, y_train),
        best_clf.score(X_test, y_test)))

    # Return best classifier model
    return best_clf

