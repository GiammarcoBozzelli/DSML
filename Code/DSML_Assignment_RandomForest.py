#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:45:54 2024

@author: timbaettig
"""

"""
DS & ML
"""
#install packages
#!pip install -U spacy

#imoprt packages
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


#import data
#directory = "/Users/timbaettig/Library/Mobile Documents/com~apple~CloudDocs/00_Privat/00_EPFL/Courses/SS 2024/Data Science and Machine Learning/Project/Data/"
training = pd.read_csv("https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/training_data.csv")
test = pd.read_csv("https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/unlabelled_test_data.csv")

#data exploration

#data preparation
def preprocess_text(text):
    # Remove punctuation, digits, etc.
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Lowercase
    text = text.lower()
    # Tokenization and remove stop words
    tokens = text.split()
    stop_words = set(stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = SnowballStemmer('french')
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

training['processed_sentence'] = training['sentence'].apply(preprocess_text)

# Feature Engineering
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(training['processed_sentence'])
y = training['difficulty']

#Model Selection & Training - Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


#------------------------------------------------------------------------------------------------
#Prediction Preparation 
test['processed_sentence'] = test['sentence'].apply(preprocess_text)

#Feature Engineering
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(training['processed_sentence'])
y_train = training['difficulty']
X_test = vectorizer.transform(test['processed_sentence'])

#Train on Full Data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
predicted_difficulties = model.predict(X_test)

# Final Dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'difficulty': predicted_difficulties})

print(submission.head())

#Export
#submission.to_csv(directory+'Outputs/prediction1_randomforest.csv', index=False)

