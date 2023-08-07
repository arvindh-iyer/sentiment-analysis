import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu,sigmoid,softmax,linear
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

f=pd.read_csv('train.tsv',sep='\t')
title=f.keys()
data=f.values
data=np.array(data)
#print(title)
#print(data)

test_data=pd.read_csv('test.tsv',sep='\t')
keys=test_data.keys()
test_data=test_data.values
test_data=np.array(test_data)
print(test_data)
sr_no=test_data[:,0]
test_phrases=test_data[:,2]
print(test_phrases)



vocab={}
s=set()
phrases=data[:,2]
sentiments=data[:,3]

sentiments=np.array(sentiments,dtype=int)
indices=np.where(np.isnan(sentiments))
sentiments[indices]=2


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

for i in range(len(phrases)):
    phrases[i]=process_tweet(phrases[i])


for phrase,sentiment in zip(phrases,sentiments):
    for word in phrase:
        if word in s:
            vocab[word][sentiment]+=1
        else:
            s.add(word)
            vocab[word]=np.zeros(5)
            vocab[word][sentiment]=1

#print(vocab)

phrase_vector=np.zeros((len(phrases),5))
for i in range(len(phrases)):
    for word in phrases[i]:
        phrase_vector[i]+=vocab[word]

#print(phrase_vector.shape)
#print(phrase_vector)

x_mean=np.mean(phrase_vector,axis=0)
x_std=np.std(phrase_vector,axis=0)
#print(x_mean.shape)
phrase_vector=(phrase_vector-x_mean)/x_std

#print(phrase_vector)

sentiments=np.reshape(sentiments,(-1,1))
print(type(sentiments))
print(type(phrase_vector))
print(sentiments.dtype)
print(phrase_vector.dtype)

l=int(.8*len(phrase_vector))
train_phrase_vector=phrase_vector[:l]
cv_phrase_vector=phrase_vector[l:]

train_sentiments=sentiments[:l]
cv_sentiments=sentiments[l:]

'''model=Sequential([
    Dense(units=100,activation=relu),
    Dense(units=50,activation=relu),
    Dense(units=10,activation=relu),
    Dense(units=5,activation=linear)
])'''

model=Sequential([
    Dense(units=1000,activation=relu),
    Dense(units=500,activation=relu),
    Dense(units=100,activation=relu),
    Dense(units=50,activation=relu),
    Dense(units=10,activation=relu),
    Dense(units=5,activation=linear)
])

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(.01)
)


model.fit(phrase_vector,sentiments,epochs=20)

for i in range(len(test_phrases)):
    print(test_phrases[i])
    test_phrases[i]=str(test_phrases[i])
    test_phrases[i]=process_tweet(test_phrases[i])

test_phrases_vector=np.zeros((len(test_phrases),5))

for i in range(len(test_phrases)):
    for word in test_phrases[i]:
        if word in vocab:
            test_phrases_vector[i]+=vocab[word]

test_phrases_vector=(test_phrases_vector-x_mean)/x_std

y_output=model.predict(test_phrases_vector)

y_test=[]
for i in y_output:
    index=np.argmax(i)
    y_test.append(index)
#print(y_output)


y_test=np.array(y_test).reshape((-1,1))
y_test=y_test.astype(int)

sr_no=np.reshape(sr_no,(-1,1))
sr_no=sr_no.astype(int)

headers='PhraseId,Sentiment'
output_file="output.csv"
np.savetxt(output_file,np.concatenate([sr_no,y_test],axis=1),delimiter=',',header=headers,comments="",fmt='%d')







'''y_output=model.predict(cv_phrase_vector)
y_final=np.zeros(len(y_output))

for i in range(len(y_output)):
    arg=np.argmax(y_output[i])
    y_final[i]=arg
y_final=np.reshape(y_final,(-1,1))

count=0
for i in range(len(y_final)):
    if y_final[i][0]!=cv_sentiments[i][0]:
        count+=1
print(count)'''









