import os
import sys
sys.path.append(os.path.abspath('..'))
import streamlit as st
import pickle
from src import config

st.set_page_config(
    page_title="Sentiment Analysis",  # Titolo della pagina
    page_icon="💻",  # Favicon
)

with open(f"{config.MODELS_PATH}vectorizer.pickle", "rb") as f:    
        vectorizer = pickle.load(f)



st.title("Text Representation")

model_name = st.selectbox(
    "Select Model",
    ("Random Forest", "Logistic Regression"),
)
model_path = f"{config.MODELS_PATH}random_forest.pickle" if model_name == "Random Forest" else f"{config.MODELS_PATH}logistic_regression.pickle"

if not os.path.exists(model_path):
    st.error(f"No trained model found for {model_name}. Run the training script first.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)  

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