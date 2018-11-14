import os
from collections import Counter
from bisect import bisect_left
from math import ceil
from scipy.special import binom
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

pos_dir = os.path.dirname(__file__) + "/data/POS-tokenized"
neg_dir = os.path.dirname(__file__) + "/data/NEG-tokenized"

ps = PorterStemmer()

###### THINGS TO CHANGE ######
presence = False
stem = False
num_folds = 10
unigrams = True
bigrams = False
feature_cutoff = 7
##############################

def calculate_p(scores1, scores2):
    Plus, Null, Minus = 0, 0, 0
    for i in range(0, 10):
        if scores1[i] > scores2[i]:
            Plus += 1
        elif scores1[i] == scores2[i]:
            Null += 1
        else:
            Minus += 1

    N = 2*ceil(Null/2) + Plus + Minus
    k = ceil(Null/2) + min(Plus, Minus)
    q = 0.5

    p = 0
    for i in range(0, k+1):
        p += binom(N, i) * pow(q, i) * pow((1 - q), N-i)
    p *= 2

    return p

def make_dictionary(train_reviews):
    all_words = []
    for review in train_reviews:
        with open(review, encoding="utf-8") as r:
            prev_word = None
            for line in r:
                word = ps.stem(line.rstrip()) if stem else line.rstrip()
                if word.isalpha() and len(word) > 1:
                    if unigrams:
                        all_words.append((word, ))
                    if prev_word:
                        if bigrams:
                            all_words.append((prev_word, word))
                else:
                    word = None
                prev_word = word
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(len(dictionary))
    dictionary = [(w, n) for (w, n) in dictionary if n > feature_cutoff]
    dictionary.sort(key = lambda tup: tup[0])
    return dictionary

def make_dictionary_presence(train_reviews):
    all_words = []
    for review in train_reviews:
        with open(review, encoding="utf-8") as r:
            prev_word = None
            seen_words = []
            for line in r:
                word = ps.stem(line.rstrip()) if stem else line.rstrip()
                if word.isalpha() and len(word) > 1 and word not in seen_words:
                    if unigrams:
                        all_words.append((word, ))
                    if prev_word:
                        if bigrams:
                            all_words.append((prev_word, word))
                    seen_words.append(word)
                else:
                    word = None
                prev_word = word
    dictionary = Counter(all_words)
    dictionary = dictionary.most_common(len(dictionary))
    dictionary = [(w, n) for (w, n) in dictionary if n > feature_cutoff]
    dictionary.sort(key = lambda tup: tup[0])
    return dictionary

def extract_features(reviews):
    features_matrix = np.zeros((len(reviews), len(dictionary)))
    docID = 0
    for review in reviews:
        with open(review, encoding="utf-8") as r:
            prev_word = None
            for line in r:
                word = ps.stem(line.rstrip()) if stem else line.rstrip()
                wordID = 0
                if unigrams:
                    i = bisect_left(dictionary, ((word, ), 0))
                    if i != len(dictionary) and dictionary[i][0] == (word, ):
                        wordID = i
                        features_matrix[docID, wordID] += 1
                if prev_word:
                    if bigrams:
                        i = bisect_left(dictionary, ((prev_word, word), 0))
                        if i != len(dictionary) and dictionary[i][0] == (prev_word, word):
                            wordID = i
                            features_matrix[docID, wordID] += 1
                prev_word = word
            docID = docID + 1 
    return features_matrix

def extract_features_presence(reviews):
    features_matrix = np.zeros((len(reviews), len(dictionary)))
    docID = 0
    for review in reviews:
        with open(review, encoding="utf-8") as r:
            prev_word = None
            for line in r:
                word = ps.stem(line.rstrip()) if stem else line.rstrip()
                wordID = 0
                if unigrams:
                    i = bisect_left(dictionary, ((word, ), 0))
                    if i != len(dictionary) and dictionary[i][0] == (word, ):
                        wordID = i
                        features_matrix[docID, wordID] = 1
                if prev_word:
                    if bigrams:
                        i = bisect_left(dictionary, ((prev_word, word), 0))
                        if i != len(dictionary) and dictionary[i][0] == (prev_word, word):
                            wordID = i
                            features_matrix[docID, wordID] = 1
                prev_word = word
            docID = docID + 1
    return features_matrix

def get_stratified_split(pos_dir, neg_dir, num_folds, offset):
    pos_reviews = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
    neg_reviews = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
    train_reviews = []
    test_reviews = []

    i = 0
    while i < len(pos_reviews):
        test_reviews.append(pos_reviews[i + offset])
        for j in range(1, num_folds):
            index = i + ((j + offset) % num_folds)
            train_reviews.append(pos_reviews[index])
        i += num_folds

    i = 0
    while i < len(neg_reviews):
        test_reviews.append(neg_reviews[i + offset])
        for j in range(1, num_folds):
            index = i + ((j + offset) % num_folds)
            train_reviews.append(neg_reviews[i + (j % num_folds)])
        i += num_folds

    return (train_reviews, test_reviews)

scores1, scores2 = [], []
for i in range(num_folds):
    print("Splitting reviews into training/test sets: " + str(i+1) + "/" + str(num_folds))
    (train_reviews, test_reviews) = get_stratified_split(pos_dir, neg_dir, num_folds, i)

    print("Building training dictionary...")
    dictionary = make_dictionary_presence(train_reviews) if presence else make_dictionary(train_reviews)

    print("Extracting training features...")
    train_length = len(train_reviews)
    train_labels = np.zeros(train_length)
    train_labels[train_length//2:train_length] = 1
    train_matrix = extract_features_presence(train_reviews) if presence else extract_features(train_reviews)

    print("Training naive bayes model...")
    model1 = MultinomialNB()
    model1.fit(train_matrix, train_labels)

    print("Training SVM model...")
    model2 = SVC(kernel="linear")
    model2.fit(train_matrix, train_labels)

    print("Extracting test features...")
    test_length = len(test_reviews)
    test_labels = np.zeros(test_length)
    test_labels[test_length//2:test_length] = 1
    test_matrix = extract_features_presence(test_reviews) if presence else extract_features(test_reviews)

    print("Gathering accuracy scores...")
    result1 = model1.predict(test_matrix)
    scores1.append(accuracy_score(test_labels, result1))
    result2 = model2.predict(test_matrix)
    scores2.append(accuracy_score(test_labels, result2))

p = calculate_p(scores1, scores2)

print("")
print("Features:" + " Unigrams" if unigrams else "" + " Bigrams" if bigrams else "")
print("No. Features: " + str(len(dictionary)))
print("Frequency or Presence?: " + "Presence" if presence else "Frequency")
print("Stemmed?: " + "Yes" if stem else "No")
print("NB Score: " + str(np.mean(scores1)))
print("SVM Score: " + str(np.mean(scores2)))
print("P-Value: " + str(p))