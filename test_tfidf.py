#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:21:40 2018

@author: BriseZoey
"""


import helper
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def test(clf, xtrain, ytrain, xtest, ytest):
    print('train score:', clf.score(xtrain, ytrain))
    print('test score:', clf.score(xtest, ytest))
    print('cv10 mean:', cross_val_score(clf, xtrain, ytrain, cv=10).mean())

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

wordindex_dict = dict()
for word, count in wordcount_list:
    wordindex_dict[word] = len(wordindex_dict)
    
## fetch train data

xtrain_words = [' '.join(words) for words in strategy_instance.class0]

vectorizer = CountVectorizer(ngram_range=(1,5))
transformer = TfidfTransformer()

vectorizer.fit(xtrain_words)
xtrain_words += [' '.join(words) for words in strategy_instance.class1]
xtrain_counts = vectorizer.transform(xtrain_words)
xtrain_tfidf = transformer.fit_transform(xtrain_counts)

xtrain = xtrain_tfidf

ytrain = [0] * len(strategy_instance.class0)
ytrain += [1] * len(strategy_instance.class1)

skb = SelectKBest(chi2, k=5720)
skb.fit(xtrain, ytrain)
xtrain = skb.transform(xtrain)

## fetch test data

with open('test_data.txt', 'r') as test_class1_file:
    xtest_words = [line for line in test_class1_file]
    xtest_counts = vectorizer.transform(xtest_words)
    xtest_tfidf = transformer.transform(xtest_counts)
    
    xtest = skb.transform(xtest_tfidf)
    
    ytest = [1] * len(xtest_words)

## training and testing
parameters={'gamma': 'auto', 'C': 1, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}

clf = strategy_instance.train_svm(parameters, xtrain, ytrain)
ypredict = clf.predict(xtest)

print(confusion_matrix(ytest, ypredict))
#print(classification_report(ytest, ypredict, target_names=['class0', 'class1']))
test(clf, xtrain, ytrain, xtest, ytest)


