# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:32:34 2025

@author: kiran
"""

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')     # Main lemmatizer corpus
nltk.download('omw-1.4')     # Required for some WordNet dependencies
nltk.download('stopwords')
nltk.download('wordnet')

# Load model, vectorizer, and encoder
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.title("Drug Review Condition Classifier")
st.write("Enter a patient review and the model will predict the condition.")

review_input = st.text_area("Enter patient review here:")

if st.button("Predict Condition"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)
        condition = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Condition: **{condition}**")