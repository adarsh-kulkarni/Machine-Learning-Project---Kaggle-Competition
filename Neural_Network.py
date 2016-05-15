# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:34:39 2015

@author: adarsh
"""
import pandas as pd
import numpy as np
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sknn.mlp import Classifier, Layer

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

tf_transformer = TfidfTransformer()
X_train = tf_transformer.fit_transform(X_train_counts)
#y_train = tf_transformer.fit_transform(y_train_counts)

test_data = pd.read_json("test.json")

test_ingredients = test_data['ingredients']
test_id = test_data['id']

test_data['ingredients_test'] = [' '.join([lmtzr.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).encode('utf-8').strip() for lists in test_ingredients]
test_words=test_data['ingredients_test']

X_test_counts = count_vect.transform(test_words).toarray()
X_test = tf_transformer.transform(X_test_counts)
#X_test = tf_transformer.fit_transform(X_test_counts)
#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
nn = Classifier(
    layers=[
        Layer("Tanh", units=300),Layer("Tanh", units=300),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=30, regularize="L2")

nn.fit(X_train_counts, cuisine)

predictions = nn.predict(X_test_counts)

words_list = [' '.join(x) for x in predictions]


submission = pd.DataFrame( data={"id":test_id, "cuisine":words_list} )

submission.to_csv( "Neural_nw_cooking.csv", index=False, quoting=3 )