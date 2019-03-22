# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
# nltk.download('stopwords')
# nltk.download('punkt')
# Importing the dataset

docs = load_files('contract/',shuffle=False)
X,y = docs.data,docs.target
print(X[0])

"""
# Storing the data as pickle file
with open('X.pickle','wb') as f:
	pickle.dump(X,f)
	
"""	
# Creating the corpus
corpus = []
for i in range(0,len(X)) :
	doc = re.sub(r'\W',' ',str(X[i]))
	doc = doc.lower()
	doc = re.sub(r' [n]',' ',doc)
	doc = re.sub(r'^[a-z]\s+',' ',doc)
	doc = re.sub(r'\s+',' ',doc)
	doc= re.sub(r"[-()\"#/@;_:<>{}+=~|.?,]","",doc)
	corpus.append(doc)
	
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features =2000, min_df =3,max_df = 0.45,stop_words = stopwords.words('english'))	
X = vectorizer.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()	
X = transformer.fit_transform(X).toarray()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Supervised approach
# Support Vector Classification using linear kernel
from sklearn.svm import SVC  
classifier1 = SVC(kernel='linear')
classifier1.fit(X_train,y_train)
y_pred = classifier1.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('Classification model metrics : ')
print('1.F1 score = ',f1_score(y_test, y_pred, average="macro"))
print('2.Precision score = ',precision_score(y_test, y_pred, average="macro"))
print('3.Recall score = ',recall_score(y_test, y_pred, average="macro")) 
print('4.Accuracy score = ',accuracy_score(y_test, y_pred))

#Unsupervised approch
# Using Kmean clustering 
from sklearn.cluster import KMeans
model = KMeans(n_clusters =2)
model.fit(X_train)

targets = model.predict(X_test)
cm1 = confusion_matrix(y_test,targets)
print('1.Accuracy score = ',accuracy_score(y_test,targets))

#Pickling the classifier and model
with open('kmean.pickle','wb') as f:
	pickle.dump(model,f)
with open('classifier.pickle','wb') as f:
	pickle.dump(classifier1,f)
	 