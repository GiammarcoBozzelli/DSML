#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:26:51 2024

@author: timbaettig
"""

#!pip3 install streamlit
#pip install streamlit pytube youtube-transcript-api
#pip install requests joblib

#imoprt packages
import streamlit as st
!pip install joblib
import joblib
import requests
import os
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')


#import data
directory = "/Users/timbaettig/Library/Mobile Documents/com~apple~CloudDocs/00_Privat/00_EPFL/Courses/SS 2024/Data Science and Machine Learning/Project/Data/"
training = pd.read_csv("https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/training_data.csv")
test = pd.read_csv("https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/unlabelled_test_data.csv")


#model implementation----------------------------------------------------------
# Load the vectorizer
url = 'https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/tfidf_vectorizer_svm.pkl'
response = requests.get(url)
open('vectorizer.pkl', 'wb').write(response.content)
vectorizer = joblib.load('vectorizer.pkl')

# Load the trained SVM model
url = 'https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/svm_model.pkl'
response = requests.get(url)
open('svm_model.pkl', 'wb').write(response.content)
svm_model = joblib.load("svm_model.pkl")

#------------------------------------------------------------------------------
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

# Function to extract video ID from YouTube URL
def extract_video_id(youtube_url):
    return YouTube(youtube_url).video_id

# Function to fetch transcript
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
    return " ".join([entry['text'] for entry in transcript])

def analyze_transcript(transcript):
    difficulty = svm_model.predict(transcript)
    return difficulty

def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    return yt.thumbnail_url
#------------------------------------------------------------------------------
# Streamlit app
st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(to right, #7aecbe, #3ce1c5);
        }
        .sidebar .sidebar-content {
            background: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.image("/Users/timbaettig/Library/Mobile Documents/com~apple~CloudDocs/00_Privat/00_EPFL/Courses/SS 2024/Data Science and Machine Learning/Project/Data/B&B_Digital.webp", width=300)

st.title("How Difficult Is My YouTube Video?")

# Use session state to store variables
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ''

if 'transcript' not in st.session_state:
    st.session_state.transcript = ''

if 'difficulty' not in st.session_state:
    st.session_state.difficulty = ''

if 'sentence_difficulties' not in st.session_state:
    st.session_state.sentence_difficulties = []

if 'sentences' not in st.session_state:
    st.session_state.sentences = []

youtube_url = st.text_input("Enter the YouTube video URL:", st.session_state.youtube_url)

if st.button("Analyze"):
    if youtube_url:
        try:
            st.session_state.youtube_url = youtube_url
            video_id = extract_video_id(youtube_url)
            transcript = get_transcript(video_id)
            sentences = transcript.split('. ')
            processed_sentences = [preprocess_text(sentence) for sentence in sentences]
            transformed_sentences = vectorizer.transform(processed_sentences)
            sentence_difficulties = analyze_transcript(transformed_sentences)
            difficulty = analyze_transcript(vectorizer.transform([preprocess_text(transcript)]))
            thumbnail_url = get_thumbnail_url(youtube_url)
            st.session_state.transcript = transcript
            st.session_state.sentences = sentences
            st.session_state.sentence_difficulties = sentence_difficulties
            st.session_state.difficulty = difficulty[0]
            st.session_state.thumbnail_url = thumbnail_url
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please enter a valid YouTube URL.")

# Conditionally render based on session state
if st.session_state.difficulty:
    st.image(st.session_state.thumbnail_url, caption='YouTube Video Thumbnail')
    st.markdown(f"The difficulty of the French spoken in this video is: **{st.session_state.difficulty}**")

    selected_difficulty = st.selectbox("Select a difficulty to explore sentences:", ["A1", "A2", "B1", "B2", "C1", "C2"])
    st.markdown(f"Sentences with difficulty {selected_difficulty}:")
    for sentence, difficulty in zip(st.session_state.sentences, st.session_state.sentence_difficulties):
        if difficulty == selected_difficulty:
            st.write(sentence)

    # Section to specify the actual level
    st.markdown("## Is this your actual level?")
    all_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    available_levels = [level for level in all_levels if level != st.session_state.difficulty]
    actual_level = st.selectbox("Select your actual level if different:", available_levels)
    if actual_level != st.session_state.difficulty:
        st.markdown(f"Curated list of videos for level {actual_level}:")
        curated_videos = {
            "A1": ["https://www.youtube.com/watch?v=A1", "https://www.youtube.com/watch?v=A2"],
            "A2": ["https://www.youtube.com/watch?v=A3", "https://www.youtube.com/watch?v=A4"],
            "B1": ["https://www.youtube.com/watch?v=B1", "https://www.youtube.com/watch?v=B2"],
            "B2": ["https://www.youtube.com/watch?v=B3", "https://www.youtube.com/watch?v=B4"],
            "C1": ["https://www.youtube.com/watch?v=C1", "https://www.youtube.com/watch?v=C2"],
            "C2": ["https://www.youtube.com/watch?v=C3", "https://www.youtube.com/watch?v=C4"]
        }
        for video_url in curated_videos[actual_level]:
            st.markdown(f"[Video]({video_url})")
