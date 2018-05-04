#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 08:42:09 2018

@author: BriseZoey
"""

import helper
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# step 1, get all word and its counts, sort by word frequency
# we get word_count_list

strategy_instance = helper.strategy()
wordcount_dict = dict()
word1count_dict = dict()
word2count_dict = dict()

for words in strategy_instance.class0:
    for word in words:
        if word in wordcount_dict:
            wordcount_dict[word] += 1
        else:
            wordcount_dict[word] = 1
        if word in word1count_dict:
            word1count_dict[word] += 1
        else:
            word1count_dict[word] = 1
            
            
for words in strategy_instance.class1:
    for word in words:
        if word in wordcount_dict:
            wordcount_dict[word] += 1
        else:
            wordcount_dict[word] = 1
        if word in word2count_dict:
            word2count_dict[word] += 1
        else:
            word2count_dict[word] = 1
            
wordcount_list = sorted(wordcount_dict.items(), key=lambda x: -x[1])


## step 2, get word and its index
# the smaller the word freq, the smaller the word index
# add 'UNK' as unknown word into the front
# we get dictionary

dictionary = dict()
dictionary['UNK'] = 0
for word, count in wordcount_list:
    dictionary[word] = len(dictionary)


    
#names = [dictionary[word] for word, count in wordcount_list]
#values = [np.log(count) for word, count in wordcount_list]
#plt.bar(range(len(wordcount_list)), values, tick_label=names)
#
#names = [dictionary[word] for word, count in wordcount_list]
#values = [np.log(word1count_dict[word]) if word in word1count_dict else 0 for word, count in wordcount_list]
#plt.bar(range(len(wordcount_list)), values, tick_label=names)
#
#names = [dictionary[word] for word, count in wordcount_list]
#values = [np.log(word2count_dict[word]) if word in word2count_dict else 0 for word, count in wordcount_list]
#plt.bar(range(len(wordcount_list)), values, tick_label=names)
#
#plt.savefig('wordcount.png')
#plt.show()
    


