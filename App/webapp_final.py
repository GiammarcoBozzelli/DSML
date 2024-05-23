#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:54:57 2024

@author: timbaettig
"""
# Import packages
import streamlit as st
import random
import numpy as np
import pandas as pd
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import torch
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import CamembertTokenizer, CamembertForSequenceClassification, FlaubertTokenizer, FlaubertForSequenceClassification

# Load pre-trained models and tokenizers
model_name = "kurthhenry/camembert"
camembert_tokenizer = CamembertTokenizer.from_pretrained(model_name, use_fast=False)
camembert_model = CamembertForSequenceClassification.from_pretrained(model_name)

flaubert_path = "kurthhenry/flaubert"
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(flaubert_path, use_fast=False)
flaubert_model = FlaubertForSequenceClassification.from_pretrained(flaubert_path)

device = 0 if torch.cuda.is_available() else -1
camembert_classifier = pipeline('text-classification', model=camembert_model, tokenizer=camembert_tokenizer, framework='pt', device=device, return_all_scores=True)
flaubert_classifier = pipeline('text-classification', model=flaubert_model, tokenizer=flaubert_tokenizer, framework='pt', device=device, return_all_scores=True)

#------------------------------------------------------------------------------
# Data preparation
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

# Function to analyze transcript
def analyze_transcript(transcript):
    sentences = transcript.split('. ')
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]

    camembert_probs = camembert_classifier(processed_sentences)
    flaubert_probs = flaubert_classifier(processed_sentences)

    camembert_probs_array = np.array([[prob['score'] for prob in probs] for probs in camembert_probs])
    flaubert_probs_array = np.array([[prob['score'] for prob in probs] for probs in flaubert_probs])

    average_probs = (camembert_probs_array + flaubert_probs_array) / 2
    final_preds = np.argmax(average_probs, axis=1)

    label_mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    final_labels = [inverse_label_mapping[pred] for pred in final_preds]

    return final_labels

def get_thumbnail_url(youtube_url):
    yt = YouTube(youtube_url)
    return yt.thumbnail_url

#------------------------------------------------------------------------------
# Recommended Video Dictionary 
videos = pd.read_csv('https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/videos.csv')
videos = videos[["url", "difficulty"]]

curated_videos = {}
for difficulty in videos['difficulty'].unique():
    curated_videos[difficulty] = videos[videos['difficulty'] == difficulty]['url'].tolist()

if np.nan in curated_videos:
    curated_videos['A1'] = curated_videos.pop(np.nan)

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

st.image("https://raw.githubusercontent.com/GiammarcoBozzelli/DSML/main/DATA/B&B_Digital.webp", width=300)

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
            sentence_difficulties = analyze_transcript(transcript)
            difficulty = analyze_transcript(transcript)[0]  # Only the first item is needed
            thumbnail_url = get_thumbnail_url(youtube_url)
            st.session_state.transcript = transcript
            st.session_state.sentences = sentences
            st.session_state.sentence_difficulties = sentence_difficulties
            st.session_state.difficulty = difficulty
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
    available_levels = [level for level in all_levels if level != st.session_state.get('difficulty', '')]
    actual_level = st.selectbox("Select your actual level if different:", available_levels)
    
    if actual_level and actual_level != st.session_state.get('difficulty', ''):
        if actual_level == "A1":
            st.markdown("There are no trending videos with difficulty A1. Please consider acquiring the very basics of the language before watching French videos.")
            duo_link = "https://www.duolingo.com/course/fr/en/Learn-French"
            st.markdown(f"This might be an option for you: {duo_link}")
        else:
            st.markdown(f"Curated list of currently trending French videos for level {actual_level}:")

            # Select two random videos
            videos_for_level = curated_videos.get(actual_level, [])
            if len(videos_for_level) > 2:
                random_videos = random.sample(videos_for_level, 2)
            else:
                random_videos = videos_for_level

            for video_url in random_videos:
                st.markdown("What about this one?")
                st.video(video_url)
