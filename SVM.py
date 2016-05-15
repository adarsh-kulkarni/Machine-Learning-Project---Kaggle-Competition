# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:34:39 2015

@author: adarsh
"""

import pandas as pd
import numpy as np
import re
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression


train_data = pd.read_json("train.json")

ingredients = train_data['ingredients']
#print ingredients

lmtzr = WordNetLemmatizer()


train_data['ingredients_string'] = [' '.join([lmtzr.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).encode('utf-8').strip() for lists in train_data['ingredients']]       
print train_data['ingredients_string']

cuisine =  train_data['cuisine']
ingredients_cuisine = train_data['ingredients_string']

count_vect = CountVectorizer(max_features=3000)
X_train_counts = count_vect.fit_transform(ingredients_cuisine).toarray()
#X_train_counts = count_vect.transform(words).toarray()


tf_transformer = TfidfTransformer()
X_train = tf_transformer.fit_transform(X_train_counts)

#print X_train,X_train.shape
print X_train[0],X_train[0].shape
print X_train[1],X_train[1].shape

test_data = pd.read_json("test.json")

test_ingredients = test_data['ingredients']
test_id = test_data['id']

test_data['ingredients_test'] = [' '.join([lmtzr.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).encode('utf-8').strip() for lists in test_ingredients]
test_words=test_data['ingredients_test']

X_test_counts = count_vect.transform(test_words).toarray()
X_test = tf_transformer.transform(X_test_counts)

#X_test = tf_transformer.fit_transform(X_test_counts)
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
#parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}

parameters = {'degree':[1,2,3,4,5,6,7]}

C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-2, 2, 5)
param_grid = dict(gamma=gamma_range, C=C_range)

clf = SVC(kernel='rbf')
#clf = LinearSVC(max_iter=2000)

classifier = grid_search.GridSearchCV(clf, param_grid)

#classifier = grid_search.GridSearchCV(clf, param_grid=param_grid)

classifier=classifier.fit(X_train,cuisine)

predictions=classifier.predict(X_test)

submission = pd.DataFrame( data={"id":test_id, "cuisine":predictions} )

submission.to_csv( "SVM_cooking.csv", index=False, quoting=3 )
