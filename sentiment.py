import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding ,LSTM , Dense,Dropout,SimpleRNN
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)
df = pd.read_csv("tweets_data.csv")
print(df)
def clean_text(text):
    text = re.sub(r'@[\w]*','',text)
    text =  re.sub(r'#[\W]*','',text)
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
df['cleaned'] = df['tweet'].apply(clean_text)
print(df)
max_words = 5000
max_len=100
tokenizer = Tokenizer(num_words=max_words,oov_token='<OOV>')
print(tokenizer)
tokenizer.fit_on_texts(df['cleaned'])
print(tokenizer)
sequences = tokenizer.texts_to_sequences(df['cleaned'])
print(sequences)
x = pad_sequences(sequences,max_len)
print(x)
y=df['label'].values
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = Sequential()
model.add(Embedding(input_dim=max_words,output_dim=64,input_length=max_len))
model.build(input_shape=(None, max_len))
model.add(LSTM(64,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train,y_train,epochs=5,batch_size=64,validation_split=0.1)
loss,acc = model.evaluate(x_test,y_test)
print(loss)
print(acc)

model.save("sentiment_lstm_model.h5")

import pickle

with open("tokenizer.pkl",'wb') as f:
    pickle.dump(tokenizer,f)
