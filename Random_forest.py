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
#y_train_counts = count_vect.fit_transform(cuisine).toarray()

tf_transformer = TfidfTransformer()
X_train = tf_transformer.fit_transform(X_train_counts)
#y_train = tf_transformer.fit_transform(y_train_counts)

randomForest = RandomForestClassifier(n_estimators=40)
clf = randomForest.fit(X_train,cuisine)

test_data = pd.read_json("test.json")

test_ingredients = test_data['ingredients']
test_id = test_data['id']

test_data['ingredients_test'] = [' '.join([lmtzr.lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).encode('utf-8').strip() for lists in test_ingredients]
test_words=test_data['ingredients_test']

X_test_counts = count_vect.transform(test_words).toarray()
X_test = tf_transformer.transform(X_test_counts)
#X_test = tf_transformer.fit_transform(X_test_counts)

cuisine_accuracy = randomForest.predict(X_test)

submission = pd.DataFrame( data={"id":test_id, "cuisine":cuisine_accuracy} )

submission.to_csv( "Random_Forest.csv", index=False, quoting=3 )