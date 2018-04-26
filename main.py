# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:17:14 2018

@author: franc
"""

import helper
import numpy as np


strategy_instance = helper.strategy()

dictionary = dict()
dictionary['UNK'] = 0
for words in strategy_instance.class0:
    for word in words:
        if word not in dictionary:
            dictionary[word] = len(dictionary)

for words in strategy_instance.class1:
    for word in words:
        if word not in dictionary:
            dictionary[word] = len(dictionary)

reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

x_train = []
y_train = []
for words in strategy_instance.class0:
    tokens = [0] * len(dictionary)
    for word in words:
        tokens[dictionary[word]] = words.count(word) ## 1
    x_train.append(tokens)
    y_train.append(0)

for words in strategy_instance.class1:
    tokens = [0] * len(dictionary)
    for word in words:
        tokens[dictionary[word]] = words.count(word) ## 1
    x_train.append(tokens)
    y_train.append(1)

x_train = np.array(x_train)
y_train = np.array(y_train)

parameters={'gamma': 'auto', 'C': 1.0, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}

#clf = strategy_instance.train_svm(parameters, x_train, y_train)
#print('training_score:', clf.score(x_train, y_train))

with open('test_data.txt', 'r') as test_class1_file:
    test_class1 = [line.strip().split(' ') for line in test_class1_file]

x_test = []
y_test = []
for words in test_class1:
    tokens = [0] * len(dictionary)
    for word in words:
        if word in dictionary:
            tokens[dictionary[word]] = words.count(word) ## 1
    x_test.append(tokens)
    y_test.append(1)

x_test = np.array(x_test)
y_test = np.array(y_test)


#print('testing_score:', clf.score(x_test, y_test))

training_data = []
for tokens in strategy_instance.class0:
    training_data += [' '.join(tokens)]

for tokens in strategy_instance.class1:
    training_data += [' '.join(tokens)]


## 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
x_train_gai = vectorizer.fit_transform(training_data)

##
clf2 = strategy_instance.train_svm(parameters, x_train_gai, y_train)
print('training_score_2:', clf2.score(x_train_gai, y_train))
# 
with open('test_data.txt', 'r') as test_class1_file:
    testing_data = [line.strip() for line in test_class1_file]
x_test_gai = vectorizer.transform(testing_data)

print('testing_score_2:', clf2.score(x_test_gai, y_test))

# =============================================================================
# model selection
# verbose 1是ppt模式，2是BB模式，
# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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
model = SVC(kernel='linear')
#param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
param_grid = {'C': range(1, 15)}

grid_search = GridSearchCV(model, param_grid, verbose=2, cv=10)
grid_search.fit(x_train_gai, y_train)

print(grid_search.best_estimator_)

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
#param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]} 
#
#grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
#grid_search.fit(x_train_gai, y_train)
#
#print(model.best_estimator_)


# =============================================================================
# kernel=sigmod
# C
# gamma
# coef0








