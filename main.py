# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:17:14 2018

@author: BriseKael
"""

import helper
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# step 1, get all word and its counts, sort by word frequency
# we get word_count_list

strategy_instance = helper.strategy()
wordcount_dict = dict()
word0count_dict = dict()
word1count_dict = dict()

for words in strategy_instance.class0:
    for word in words:
        if word in wordcount_dict:
            wordcount_dict[word] += 1
        else:
            wordcount_dict[word] = 1
        if word in word0count_dict:
            word0count_dict[word] += 1
        else:
            word0count_dict[word] = 1

for words in strategy_instance.class1:
    for word in words:
        if word in wordcount_dict:
            wordcount_dict[word] += 1
        else:
            wordcount_dict[word] = 1
        if word in word1count_dict:
            word1count_dict[word] += 1
        else:
            word1count_dict[word] = 1
            
wordcount_list = sorted(wordcount_dict.items(), key=lambda x: [-x[1], x[0]])

## step 2, get word and its index
# the smaller the word freq, the smaller the word index
# add 'UNK' as unknown word into the front
# we get dictionary

wordindex_dict = dict()
wordindex_dict['CLASS0_'] = 0
wordindex_dict['CLASS1_'] = 1
wordindex_dict['UNK'] = 2
for word, count in wordcount_list:
    if count == 1:
        continue
    wordindex_dict[word] = len(wordindex_dict)
    

## step 3, x_train is the train data with word index sort by freq
# and its counts in this sentence
# idx   [0, 1, 2, ..., len(dictionary)]
# count [0, 9, 5, ..., 0, 0, 0]
# we get x_train, y_train as np.array

xtrain = []
ytrain = []
for words in strategy_instance.class0:
    tokens = [0] * len(wordindex_dict)
    for word in words:
        if word in wordindex_dict:
            tokens[wordindex_dict[word]] = words.count(word) ## 1
        else:
            if word in word0count_dict:
                tokens[wordindex_dict['CLASS0_']] += 1
            if word in word1count_dict:
                tokens[wordindex_dict['CLASS1_']] += 1
            tokens[wordindex_dict['UNK']] += 1
    xtrain.append(tokens)
    ytrain.append(0)

for words in strategy_instance.class1:
    tokens = [0] * len(wordindex_dict)
    for word in words:
        if word in wordindex_dict:
            tokens[wordindex_dict[word]] = words.count(word) ## 1
        else:
            if word in word0count_dict:
                tokens[wordindex_dict['CLASS0_']] += 1
            if word in word1count_dict:
                tokens[wordindex_dict['CLASS1_']] += 1
            tokens[wordindex_dict['UNK']] += 1
    xtrain.append(tokens)
    ytrain.append(1)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

## step 5, train with xtrain, ytrain
# 

parameters={'gamma': 'auto', 'C': 0.005, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}

clf = strategy_instance.train_svm(parameters, xtrain, ytrain)
print('training_score:', clf.score(xtrain, ytrain))

# step 6, get x_test, y_test
#

with open('test_data.txt', 'r') as test_class1_file:
    test_class1 = [line.strip().split(' ') for line in test_class1_file]

xtest = []
ytest = []
for words in test_class1:
    tokens = [0] * len(wordindex_dict)
    for word in words:
        if word in wordindex_dict:
            tokens[wordindex_dict[word]] = words.count(word) ## 1
        else:
            if word in word0count_dict:
                tokens[wordindex_dict['CLASS0_']] += 1
            if word in word1count_dict:
                tokens[wordindex_dict['CLASS1_']] += 1
            tokens[wordindex_dict['UNK']] += 1
    xtest.append(tokens)
    ytest.append(1)


xtest = np.array(xtest)
ytest = np.array(ytest)

print('testing_score:', clf.score(xtest, ytest))
print('cv10 mean:', cross_val_score(clf, xtrain, ytrain).mean())



