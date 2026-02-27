import streamlit as st
import re
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

model = tf.keras.models.load_model("sentiment_lstm_model.h5")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

max_length =100
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'@[\w]*','',text)
    text =  re.sub(r'#[\W]*','',text)
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = text.lower()
    tokens = text.split()
    return ' '.join(tokens)

def preprocess(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    paded = pad_sequences(seq,maxlen = max_length)
    return paded




#Streamlit UI
st.title("sentiment Analysis using LSTM")
st.write("Enter the tweet to predict whether it is Positive or Negative")

user_input = st.text_area("Tweet the text")



if st.button("Predict"):
    if user_input.strip() =="":
        st.warning("Please enter the text")
    else:   
        processed = preprocess(user_input)
        prediction = model.predict(processed)[0][0]

        st.write("Raw prediction value:", prediction)

        sentiment = "Positive" if prediction >= 0.5 else "Negative"

        st.subheader(f"Sentiment : {sentiment}")
        st.write(f"Confidence Score : {float(prediction):.3f}")