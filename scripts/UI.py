import os
import sys
sys.path.append(os.path.abspath('..'))
import streamlit as st
import pickle
from src import config

 

with open(f"{config.MODELS_PATH}vectorizer.pickle", "rb") as f:    
        vectorizer = pickle.load(f)

with open(f"{config.MODELS_PATH}random_forest.pickle", "rb") as file:        # serve ad aprire il file e poi chiuderlo
        model = pickle.load(file)

st.title("Text Representation")

user_input = st.text_area("Enter text to classify", "")

if st.button("Classify"):
    if user_input.strip() == "":
           st.warning("Please enter some text.")
    else:
        # transform input and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        if prediction == 'positive':
              st.success(f"Predicted class: {prediction}")
        elif prediction == 'negative':
              st.warning(f"Predicted class: {prediction}")