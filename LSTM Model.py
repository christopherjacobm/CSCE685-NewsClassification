import pandas as pd
import numpy as np
import sys, os
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Input, Embedding, Dense, Dropout, GlobalMaxPool1D, Activation, Flatten
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from nltk.corpus import stopwords
from time import time

def readData():
    news_df = pd.read_csv("uci-news-aggregator.csv")
    return news_df

def removeStopWords(x):
    stopSet = set(stopwords.words('english'))

    for i in range(len(x)):
        wordList = x[i].split(" ")
        cleanLine = [word for word in wordList if word not in stopSet]
        x[i] = ' '.join(cleanLine)
    
    return x
  
def splitData(news_df):
    news_df = news_df.replace({'CATEGORY': {'b': 0, 't': 1, 'e':2, 'm':3}})
    
    x = news_df['TITLE'].values
    y = news_df['CATEGORY'].values
 
    y_one_hot_vector = to_categorical(y)

    x = removeStopWords(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot_vector, test_size=0.20, random_state=42)
    
    return x_train, x_test, y_train, y_test
    
  
def preProcess(x_train, x_test, y_train, y_test):

    max_features = 10000

    max_len = 70
  
    tokenizer = Tokenizer(num_words=max_features)
    
    # fit the tokenizer on the training data
    tokenizer.fit_on_texts(x_train)
    
    # convert text to tokens
    train_tokenized = tokenizer.texts_to_sequences(x_train)
    test_tokenized = tokenizer.texts_to_sequences(x_test)

    # keep all the news headlines to be of the same length
    x_training = pad_sequences(train_tokenized, maxlen=max_len)
    x_testing = pad_sequences(test_tokenized, maxlen=max_len)
    
    return max_features, max_len, x_training, x_testing
    
 
def getModel(max_len, max_features):
    # use learned embedding of size 128
    embedding_size = 128
    
    inpt = Input(shape=(max_len, ))
    
    # first layer as an Embedding layer
    x = Embedding(max_features, embedding_size)(inpt)
    
    # LSTM layer, nodes = 50
    x = LSTM(50, return_sequences=True, name='LSTM_layer')(x)     
 
    # max pooling
    x = GlobalMaxPool1D()(x)
    
    # use dropout for regularization
    x = Dropout(0.2)(x)
    
    # fully connected layer
    x = Dense(30, activation='relu')(x)
    
    x = Dropout(0.1)(x) 
    
    x= Dense(4, activation='softmax')(x)
    
    model = Model(inputs=inpt, outputs=x)
    
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    
    return model
    
 
def createModel(max_len, max_features, x_training, y_train):
    model = getModel(max_len, max_features)
    batch_size = 64
    epochs = 2
    model.fit(x_training, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    model_file_path = 'lstm_model.h5'
    model.save(model_file_path)
    return model
    
def main():
    news_df = readData()
    x_train, x_test, y_train, y_test = splitData(news_df)
    max_features, max_len, x_training, x_testing = preProcess(x_train, x_test, y_train, y_test)
    model = createModel(max_len, max_features, x_training, y_train)
 
if __name__ == '__main__':
    main()
