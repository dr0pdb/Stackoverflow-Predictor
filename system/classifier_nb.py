# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 02:18:39 2018

@author: srv_twry

"""

"""
    Initial setup.
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset_size = 5000
dataset = pd.read_csv('dataset/dataset.csv')
dataset = dataset.iloc[0:dataset_size,:]

# Checking for missing values
dataset.isnull().any()

"""
   Data Preprocessing. 
"""
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Cleaning the title
title_corpus = []
for i in range(0, dataset_size):
    title = re.sub('[^a-zA-Z]', ' ', dataset['Title'][i])
    title = title.lower()
    title = title.split()
    ps = PorterStemmer()
    title = [ps.stem(word) for word in title if not word in set(stopwords.words('english'))]
    title = ' '.join(title)
    title_corpus.append(title)

# Cleaning the body
body_corpus = []
for i in range(0, dataset_size):
    body = re.sub('[^a-zA-Z]', ' ', dataset['BodyMarkdown'][i])
    body = body.lower()
    body = body.split()
    ps = PorterStemmer()
    body = [ps.stem(word) for word in body if not word in set(stopwords.words('english'))]
    body = ' '.join(body)
    body_corpus.append(body)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X_title = cv.fit_transform(title_corpus).toarray()
X_body = cv.fit_transform(body_corpus).toarray()
X = np.concatenate((X_title, X_body, dataset.iloc[:,4:6]), axis = 1)
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


"""
    Training and Testing.
"""
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Getting the percentage correct predictions
np.mean(y_pred == y_test)
    

    
    





