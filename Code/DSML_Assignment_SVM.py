#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:15:38 2024

@author: timbaettig
"""

#install packages
#!pip install -U spacy
#pip install joblib


#imoprt packages
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import joblib
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



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

# Vectorize the training data
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(training['processed_sentence'])
y = training['difficulty']

# Train SVM model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_dist = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf', 'poly']),
    'degree': Integer(2, 5),   # Only relevant for 'poly' kernel
    'gamma': Categorical(['scale', 'auto'])  # Only relevant for 'rbf' and 'poly' kernels
}

# Initialize the BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=SVC(probability=True, random_state=42), 
    search_spaces=param_dist, 
    n_iter=32,  # Number of parameter settings sampled
    cv=5,       # Number of folds in cross-validation
    random_state=42,
    n_jobs=-1   # Use all available cores
)

# Fit the model
bayes_search.fit(X_train, y_train)

# Get the best model
best_svm_model = bayes_search.best_estimator_

# Predict and evaluate
y_pred = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])

# Print results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)
print(f"Best parameters: {bayes_search.best_params_}")



#------------------------------------------------------------------------------------------------
#Prediction Preparation 
test['processed_sentence'] = test['sentence'].apply(preprocess_text)

# Vectorize the test data
X_test = vectorizer.transform(test['processed_sentence'])

# Make predictions
predicted_difficulties = best_svm_model.predict(X_test)

# Save the vectorizer
#vectorizer_path = directory + "tfidf_vectorizer_svm.pkl"
#joblib.dump(vectorizer, vectorizer_path)

# Save the trained SVM model
#svm_model_path = directory + "svm_model.pkl"
#joblib.dump(best_svm_model, svm_model_path)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'difficulty': predicted_difficulties
})

# Export to CSV
#submission.to_csv(directory+'Outputs/prediction3_log_reg_svm.csv', index=False)
