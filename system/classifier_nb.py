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
dataset_size = 10000
dataset = pd.read_csv('dataset/dataset.csv')
dataset = dataset.iloc[0:dataset_size,:]

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
    
    





