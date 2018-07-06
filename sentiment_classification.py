# Sentiment analysis of collection of 100000 reviews of IMDB
# Goal: prediction of user's opinion based on her comments


import sys
import scipy
import os
import re
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn import naive_bayes, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV


os.chdir("/home/valarmorgulis/Desktop/speech-nlp/")

vector_type = "Word2vec"
model_name = "GoogleNews-vectors-negative300.bin"
model_type = "bin"
num_features = 300    
min_word_count = 40   
num_workers = 4       
context = 10         
downsampling = 1e-3

# training_model {NB", "SVM"}
training_model = "SVM"

# Note: Scaling is needed for SVM
scaling = "standard"

# dimension reduction{chi2", "no"}

dim_reduce = "no"
num_dim = 300

##################### End of Initialization #####################



##################### Function Definition #####################

def clean_review(raw_review, remove_stopwords = False, output_format = "string"):
    # Remove HTML markup
    text = BeautifulSoup(raw_review,"html.parser")

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Split words and store to list
    text = text.lower().split()

    if remove_stopwords:

        # Use set as it has O(1) lookup time
        stops = set(stopwords.words("english"))
        words = [w for w in text if w not in stops]

    else:
        words = text

    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(words)

    elif output_format == "list":
        return words


def review_to_doublelist(review, tokenizer, remove_stopwords = False):
  
    # Create a list of sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    sentence_list = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentence_list.append(clean_review(raw_sentence, False, "list"))
    return sentence_list


def review_to_vec(words, model, num_features):
   

    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0

    # index2word is a list consisting of all words in the vocabulary
    # Convert list to set for speed
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            word_count += 1
            feature_vec += model[word]

    feature_vec /= word_count
    return feature_vec


def gen_review_vecs(reviews, model, num_features):

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

       if curr_index%1000 == 0.:
           print ("Vectorizing review %d of %d" % (curr_index, len(reviews)))

       review_feature_vecs[curr_index] = review_to_vec(review, model, num_features)
       curr_index += 1

    return review_feature_vecs


##################### End of Function Definition #####################



########################### Main Program ###########################

train_list = []
test_list = []
word2vec_input = []
pred = []

train_data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=0)
test_data = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=0)

if vector_type == "Word2vec":
    unlab_train_data = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


# Extract words from reviews
if vector_type == "Word2vec" or vector_type == "Word2vec_pretrained":

    for i in range(0, len(train_data.review)):

        if vector_type == "Word2vec":
            # Decode utf-8 coding first
            word2vec_input.extend(review_to_doublelist(train_data.review[i], tokenizer))

        train_list.append(clean_review(train_data.review[i], output_format="list"))
        if i%1000 == 0:
            print ("Cleaning training review", i)

    if vector_type == "Word2vec":
        for i in range(0, len(unlab_train_data.review)):
            word2vec_input.extend(review_to_doublelist(unlab_train_data.review[i], tokenizer))
            if i%1000 == 0:
                print ("Cleaning unlabeled training review", i)

    for i in range(0, len(test_data.review)):
        test_list.append(clean_review(test_data.review[i], output_format="list"))
        if i%1000 == 0:
            print ("Cleaning test review", i)

elif vector_type != "no":
    for i in range(0, len(train_data.review)):

        train_list.append(clean_review(train_data.review[i]))
        if i%1000 == 0:
            print ("Cleaning training review", i)

    for i in range(0, len(test_data.review)):

        test_list.append(clean_review(test_data.review[i]))
        if i%1000 == 0:
            print ("Cleaning test review", i)


# Generate vectors from words
if vector_type == "Word2vec_pretrained" or vector_type == "Word2vec":

    if vector_type == "Word2vec_pretrained":
        print ("Loading the pre-trained model")
        if model_type == "bin":
            model = word2vec.Word2Vec.load_word2vec_format(model_name, binary=True)
        else:
            model = word2vec.Word2Vec.load(model_name)

    if vector_type == "Word2vec":
        print ("Training word2vec word vectors")
        model = word2vec.Word2Vec(word2vec_input, workers=num_workers, \
                                size=num_features, min_count = min_word_count, \
                                window = context, sample = downsampling)

        model.init_sims(replace=True)

        # Save the model for later use
        model.save(model_name)

    print ("Vectorizing training review")
    train_vec = gen_review_vecs(train_list, model, num_features)
    print ("Vectorizing test review")
    test_vec = gen_review_vecs(test_list, model, num_features)


elif vector_type != "no":
    if vector_type == "TFIDF":
        # Unit of gram is "word", only top 5000/10000 words are extracted
        count_vec = TfidfVectorizer(analyzer="word", max_features=10000, ngram_range=(1,2), sublinear_tf=True)

    elif vector_type == "Binary" or vector_type == "Int":
        count_vec = CountVectorizer(analyzer="word", max_features=10000, \
                                    binary = (vector_type == "Binary"), \
                                    ngram_range=(1,2))

    # Return a scipy sparse term-document matrix
    print ("Vectorizing input texts")
    train_vec = count_vec.fit_transform(train_list)
    test_vec = count_vec.transform(test_list)


# Dimemsion Reduction
if dim_reduce == "SVD":
    print ("Performing dimension reduction")
    svd = TruncatedSVD(n_components = num_dim)
    train_vec = svd.fit_transform(train_vec)
    test_vec = svd.transform(test_vec)
    print ("Explained variance ratio =", svd.explained_variance_ratio_.sum())

elif dim_reduce == "chi2":
    print ("Performing feature selection based on chi2 independence test")
    fselect = SelectKBest(chi2, k=num_dim)
    train_vec = fselect.fit_transform(train_vec, train_data.sentiment)
    test_vec = fselect.transform(test_vec)

# Transform into numpy arrays
if "numpy.ndarray" not in str(type(train_vec)):
    train_vec = train_vec.toarray()
    test_vec = test_vec.toarray()


# Feature Scaling
if scaling != "no":

    if scaling == "standard":
        scaler = preprocessing.StandardScaler()
    else:
        if scaling == "unsigned":
            scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        elif scaling == "signed":
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    print ("Scaling vectors")
    train_vec = scaler.fit_transform(train_vec)
    test_vec = scaler.transform(test_vec)


# Model training
if training_model == "RF" or training_model == "BT":

    rfc = RFC(n_estimators = 100, oob_score = True, \
              max_features = (None if training_model=="BT" else "auto"))
    print ("Training %s" % ("Random Forest" if training_model=="RF" else "bagged tree"))
    rfc = rfc.fit(train_vec, train_data.sentiment)
    print ("OOB Score =", rfc.oob_score_)
    pred = rfc.predict(test_vec)

elif training_model == "NB":
    nb = naive_bayes.MultinomialNB()
    cv_score = cross_val_score(nb, train_vec, train_data.sentiment, cv=10)
    print ("Training Naive Bayes")
    print ("CV Score = ", cv_score.mean())
    nb = nb.fit(train_vec, train_data.sentiment)
    pred = nb.predict(test_vec)

elif training_model == "SVM":
    svc = svm.LinearSVC()
    param = {'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
    print ("Training SVM")
    svc = GridSearchCV(svc, param, cv=10)
    svc = svc.fit(train_vec, train_data.sentiment)
    pred = svc.predict(test_vec)
    print ("Optimized parameters:", svc.best_estimator_)
    print ("Best CV score:", svc.best_score_)

# Output the results
if write_to_csv:
    output = pd.DataFrame(data = {"id": test_data.id, "sentiment": pred})
    output.to_csv("submission.csv", index=False)

