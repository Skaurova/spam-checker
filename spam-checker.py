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
        "Congratulations! You've won a free iPhone! Click here to claim.",
        "Hi John, are we still meeting at 5 pm today?",
        "Urgent: Your bank account is at risk. Please verify your credentials.",
        "Can you send me the report by tomorrow?",
        "Free vacation to the Bahamas! Call now to claim your prize.",
        "Hello, I hope you're doing well. Let's catch up soon."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

df['cleaned_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

vector = TfidfVectorizer()
X_train_tfidf = vector.fit_transform(X_train)
X_test_tfidf = vector.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)