'''review preprocessing'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def negate(text):
    negation = False
    result = []
    prev = None
    pprev = None
    for word in text:
        negated = "not_" + word if negation else word
        result.append(negated)
        

        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = True
        else:
            negation=False
    return result
# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter="\t",quoting=3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
from autocorrect import spell
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',dataset.values[i,0])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=negate(review)
    review=[(ps.stem(word)) for word in review if not word in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
lm=cv.vocabulary_

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





