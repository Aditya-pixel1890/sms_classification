import streamlit as st
import pickle
import sklearn
from sklearn.naive_bayes import  GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import nltk
from nltk.corpus import stopwords
import string


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tfidf=pickle.load(open('vectorier.pkl', 'rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classifier")
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # if is alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
input_sms=st.text_input("Enter the SMS")
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    # 2 vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4..display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")





