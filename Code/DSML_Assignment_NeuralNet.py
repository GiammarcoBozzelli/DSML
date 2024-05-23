#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:21:36 2024

@author: timbaettig
"""
#SETUP
# ------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization
from keras_tuner import RandomSearch
from sklearn.preprocessing import LabelEncoder

directory = "/Users/timbaettig/Library/Mobile Documents/com~apple~CloudDocs/00_Privat/00_EPFL/Courses/SS 2024/Data Science and Machine Learning/Project/Data/"
training = pd.read_csv("https://raw.githubusercontent.com/Kurthhenry/DSML/main/DATA/training_data.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Kurthhenry/DSML/main/DATA/unlabelled_test_data.csv")

# DATA PROCESSING
# ------------------------------------------------------------------------------------------------
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

# Apply preprocessing
training['processed_sentence'] = training['sentence'].apply(preprocess_text)

# Tokenization and sequence padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training['processed_sentence'])
X = tokenizer.texts_to_sequences(training['processed_sentence'])

# Pad sequences to ensure uniform input size
X = pad_sequences(X, maxlen=75)

# Checking unique difficulty levels
unique_classes = training['difficulty'].unique()
num_classes = len(unique_classes)

print("Unique classes:", unique_classes)
print("Number of classes:", num_classes)

# Convert labels to one-hot encoding
encoder = LabelEncoder()
y = encoder.fit_transform(training['difficulty'])
y = to_categorical(y, num_classes=num_classes)

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# BUILDING THE NEURAL NET
# ------------------------------------------------------------------------------------------------
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                        output_dim=hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
                        input_length=75))  # Adjusted input length as per padded sequences
    model.add(LSTM(hp.Int('lstm_units', min_value=64, max_value=256, step=64)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))  # Make sure num_classes matches the number of labels

    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------------------------
tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='my_dir',
    project_name='keras_tuning',
    overwrite=True
)

tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

best_nn_model = tuner.get_best_models(num_models=1)[0]


# Save the trained SVM model
nn_model_path = directory + "nn_model.pkl"
joblib.dump(best_nn_model, nn_model_path)

# Evalutaion
# ------------------------------------------------------------------------------------------------
# Predict and evaluate
y_test = np.argmax(y_test, axis=1)
y_test = encoder.inverse_transform(y_test)

y_pred = best_nn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_labels = encoder.inverse_transform(y_pred_classes)

accuracy = accuracy_score(y_test, y_pred_labels)
conf_matrix = confusion_matrix(y_test, y_pred_labels)
report = classification_report(y_test, y_pred_labels, target_names=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])

# Print results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)


# Prediction
# ------------------------------------------------------------------------------------------------
test['processed_sentence'] = test['sentence'].apply(preprocess_text)
X_test_final = tokenizer.texts_to_sequences(test['processed_sentence'])
X_test_final = pad_sequences(X_test_final, maxlen=75)
predictions = best_nn_model.predict(X_test_final)
predicted_classes = np.argmax(predictions, axis=1)

# Convert class indices back to original labels
predicted_labels = encoder.inverse_transform(predicted_classes)

submission = pd.DataFrame({
    'id': test['id'],
    'difficulty': predicted_labels
})

# Export to CSV
#submission.to_csv(directory + 'prediction_deep_learning_2.csv', index=False)
