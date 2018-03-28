''' Contains all the utility functions
	Will contain most of the code of predictor
'''

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier


class Question(object):

	"""docstring for Question"""
	def __init__(self, title, reputation, deleted_questions, body):
		super(Question, self).__init__()
		self.title = title
		self.reputation = reputation
		self.deleted_questions = deleted_questions
		self.body = body


	def __str__(self):
		return "title: " + self.title + "body: " + self.body


	# The function used to predict the outcome
	def predict_outcome(self):
		X_title = self.clearTitle()
		X_body = self.clearBody()
		reputation_array = np.asarray(self.reputation) 
		deleted_questions_array = np.asarray(self.deleted_questions)

		# load serialised data	
		reputation_full = joblib.load('mainapp/reputation.sav')
		deleted_full = joblib.load('mainapp/deleted.sav')
		reputation_full = np.concatenate((reputation_full, np.reshape(reputation_array, (1,1))), axis = 0)
		deleted_full = np.concatenate((deleted_full, np.reshape(deleted_questions_array, (1,1))), axis = 0)
		X = np.concatenate((X_title, X_body, reputation_full, deleted_full), axis = 1)
		X_final = self.applyFeatureScaling(X)
		y = joblib.load('mainapp/y.sav')
		
		classifier = RandomForestClassifier(n_estimators = 500, criterion = 'gini')
		classifier.fit(X_final[:-1,:], y)
		return classifier.predict(np.reshape((X_final[-1,:]), (1, -1)))


	def clearTitle(self):
		title_corpus = joblib.load('mainapp/title_corpus.sav')
		temp_title = re.sub('[^a-zA-Z]', ' ', self.title)
		temp_title = temp_title.lower()
		temp_title = temp_title.split()
		ps = PorterStemmer()
		temp_title = [ps.stem(word) for word in temp_title if not word in set(stopwords.words('english'))]
		temp_title = ' '.join(temp_title)
		title_corpus.append(temp_title)
		cv = CountVectorizer(max_features = 500)
		return cv.fit_transform(title_corpus).toarray()

	def clearBody(self):
		body_corpus = joblib.load('mainapp/body_corpus.sav')
		temp_body = re.sub('[^a-zA-Z]', ' ', self.body)
		temp_body = temp_body.lower()
		temp_body = temp_body.split()
		ps = PorterStemmer()
		temp_body = [ps.stem(word) for word in temp_body if not word in set(stopwords.words('english'))]
		temp_body = ' '.join(temp_body)
		body_corpus.append(temp_body)
		cv = CountVectorizer(max_features = 1000)
		return cv.fit_transform(body_corpus).toarray()

	def applyFeatureScaling(self, X):
		sc = StandardScaler(with_mean = True, with_std = True)
		return sc.fit_transform(X)	
		
 
