import os
from collections import Counter

from bisect import bisect_left

from nltk.stem import PorterStemmer

ps = PorterStemmer()

def make_dictionary(train_dir):
    reviews = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]    
    all_words = []
    print("Building dictionary...")      
    for review in reviews:
        with open(review, encoding="utf-8") as r:
            for line in r:
                word = ps.stem(line.rstrip())
                all_words.append(word)
    dictionary = Counter(all_words)

    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    dictionary.sort(key = lambda tup: tup[0])

    return dictionary

def extract_features(train_dir):
    reviews = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]    
    features_matrix = np.zeros((len(reviews), 3000))
    docID = 0
    for review in reviews:
        print("Extracting features for review " + str(docID) + "...")
        with open(review, encoding="utf-8") as r:
            for line in r:
                word = ps.stem(line.rstrip())
                wordID = 0
                i = bisect_left(dictionary, (word, 0))
                if i != len(dictionary) and dictionary[i][0] == word:
                    wordID = i
                    features_matrix[docID, wordID] += 1
            docID = docID + 1 
    return features_matrix

import numpy as np
np.set_printoptions(threshold = np.nan)

all_dir = os.path.dirname(__file__) + "/data/ALL-tokenized"
pos_dir = os.path.dirname(__file__) + "/data/POS-tokenized"
neg_dir = os.path.dirname(__file__) + "/data/NEG-tokenized"

dictionary = make_dictionary(all_dir)

train_labels = np.zeros(2000)
train_labels[1000:2000] = 1

pos_train_matrix = extract_features(pos_dir)
neg_train_matrix = extract_features(neg_dir)
train_matrix = np.concatenate((pos_train_matrix, neg_train_matrix), axis=0)

from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB()

from sklearn.svm import SVC
model2 = SVC()

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(model1, train_matrix, train_labels, cv=10)
print(scores1)
scores2 = cross_val_score(model2, train_matrix, train_labels, cv=10)
print(scores2)

from math import ceil
Plus, Null, Minus = 0, 0, 0
for i in range(0, 10):
    if scores1[i] > scores2[i]:
        Plus += 1
    elif scores1[i] == scores2[i]:
        Null += 1
    else:
        Minus += 1
N = 2*ceil(Null/2) + Plus + Minus