import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):

    tokens = word_tokenize(text.lower())

    stopWords = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stopWords]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

data = {
    'text': [
        "Congratulations!"
    ],
    'label': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)