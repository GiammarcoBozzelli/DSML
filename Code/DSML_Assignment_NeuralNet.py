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
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from sklearn.preprocessing import LabelEncoder

directory = "/Users/timbaettig/Library/Mobile Documents/com~apple~CloudDocs/00_Privat/00_EPFL/Courses/SS 2024/Data Science and Machine Learning/Project/Data/"
training = pd.read_csv(directory + "training_data.csv")
test = pd.read_csv(directory + "unlabelled_test_data.csv")

# DATA PROCESSING
# ------------------------------------------------------------------------------------------------
def preprocess_text(text):
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    text = text.lower()
    return text

# Apply preprocessing
training['processed_sentence'] = training['sentence'].apply(preprocess_text)
test['processed_sentence'] = test['sentence'].apply(preprocess_text)

# Tokenization and sequence padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(training['processed_sentence'])
X_train = tokenizer.texts_to_sequences(training['processed_sentence'])
X_test = tokenizer.texts_to_sequences(test['processed_sentence'])

# Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=75)
X_test = pad_sequences(X_test, maxlen=75)

# Checking unique difficulty levels
unique_classes = training['difficulty'].unique()
num_classes = len(unique_classes)

print("Unique classes:", unique_classes)
print("Number of classes:", num_classes)

# Convert labels to one-hot encoding
encoder = LabelEncoder()
y_train = encoder.fit_transform(training['difficulty'])
y_train = to_categorical(y_train, num_classes=num_classes)


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
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='keras_tuning'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

best_model = tuner.get_best_models(num_models=1)[0]

predictions = best_model.predict(X_test)


# RESULTS
# ------------------------------------------------------------------------------------------------
predicted_classes = np.argmax(predictions, axis=1)  # Get the class index with the highest probability

# Convert class indices back to original labels
predicted_labels = encoder.inverse_transform(predicted_classes)

submission = pd.DataFrame({
    'id': test['id'],
    'difficulty': predicted_labels
})

# Export to CSV
submission.to_csv(directory + 'prediction_deep_learning_2.csv', index=False)
