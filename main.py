# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:17:14 2018

@author: franc
"""

import helper
import numpy as np

# step 1, get all word and its counts, sort by word frequency
# we get word_count_list

strategy_instance = helper.strategy()
word_count_dict = dict()

for words in strategy_instance.class0:
    for word in words:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
            
word_count_list = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)

## step 2, get word and its index
# the smaller the word freq, the smaller the word index
# add 'UNK' as unknown word into the front
# we get dictionary

dictionary = dict()
dictionary['UNK'] = 0
for word, count in word_count_list:
    dictionary[word] = len(dictionary)
    
#### step 1-2, testing sort freq does matter
#### it does matter. 
#    
#dictionary = dict()
#dictionary['UNK'] = 0
#
#for words in strategy_instance.class0:
#    for word in words:
#        if word not in dictionary:
#            dictionary[word] = len(dictionary)
#
#            
#for words in strategy_instance.class1:
#    for word in words:
#        if word not in dictionary:
#            dictionary[word] = len(dictionary)

### step 3, x_train is the train data with word index sort by freq
# and its counts in this sentence
# idx   [0, 1, 2, ..., len(dictionary)]
# count [0, 9, 5, ..., 0, 0, 0]
# we get x_train, y_train as np.array

x_train = []
y_train = []
for words in strategy_instance.class0:
    tokens = [0] * len(dictionary)
    for word in words:
        if word in dictionary:
            tokens[dictionary[word]] = words.count(word) ## 1
        else:
            tokens[dictionary['UNK']] += 1
    x_train.append(tokens)
    y_train.append(0)

for words in strategy_instance.class1:
    tokens = [0] * len(dictionary)
    for word in words:
        if word in dictionary:
            tokens[dictionary[word]] = words.count(word) ## 1
        else:
            tokens[dictionary['UNK']] += 1
    x_train.append(tokens)
    y_train.append(1)

x_train = np.array(x_train)
y_train = np.array(y_train)

## step 5, train with x_train, y_train
# 

parameters={'gamma': 'auto', 'C': 1, 'kernel': 'rbf', 'degree': 3, 'coef0': 0.0}

clf = strategy_instance.train_svm(parameters, x_train, y_train)
print('training_score:', clf.score(x_train, y_train))

# step 6, get x_test, y_test
#

with open('test_data.txt', 'r') as test_class1_file:
    test_class1 = [line.strip().split(' ') for line in test_class1_file]

x_test = []
y_test = []
for words in test_class1:
    tokens = [0] * len(dictionary)
    for word in words:
        if word in dictionary:
            tokens[dictionary[word]] = words.count(word) ## 1
        else:
            tokens[dictionary['UNK']] += 1
    x_test.append(tokens)
    y_test.append(1)

x_test = np.array(x_test)
y_test = np.array(y_test)

print('testing_score:', clf.score(x_test, y_test))

# =============================================================================
# model selection
# verbose 1是ppt模式，2是BB模式，
# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# =============================================================================
# parameters={'gamma': 'auto', 'C': 1.0, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}
# ‘linear’:线性核函数
# ‘poly’:多项式核函数
# ‘rbf’:径像核函数/高斯核
# ‘sigmod’:sigmod核函数
# 
# =============================================================================
# kernel=linear
# C=10, test_score=0.45
#model = SVC(kernel='linear')
##param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
##param_grid = {'C': [0.01, 0.1, 1, 10, 100]} => 0.01
##param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1]} => 0.001
#param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
#
#grid_search = GridSearchCV(model, param_grid, verbose=2, cv=10)
#grid_search.fit(x_train, y_train)
#
#print(grid_search.best_estimator_)



# =============================================================================
# kernel=poly
# C
# degree
# gamma
# coef0
#model = SVC(kernel='poly')
#param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 
#              'gamma': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]} 
#
#grid_search = GridSearchCV(model, param_grid, n_jobs=4, verbose=1, cv=10)
#grid_search.fit(x_train_gai, y_train)
#
#print(model.best_estimator_)

# =============================================================================
# kernel=rbf
# C
# gamma

#model = SVC(kernel='rbf')
#param_grid = {'C': [0.1, 1, 10, 100], 
#              'gamma': [0.1, 1, 10, 100]} 
#
#grid_search = GridSearchCV(model, param_grid, verbose=2, cv=5)    
#grid_search.fit(x_train, y_train)
#
#print(grid_search.best_estimator_)


# =============================================================================
# kernel=sigmod
# C
# gamma
# coef0


