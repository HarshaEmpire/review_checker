#importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# function to negate a word
def negate(text):
    negation = False
    result = []
    prev = None
    pprev = None
    for word in text:
        negated = "not_" + word if (negation and word not in set(stopwords.words('english'))) else word
        result.append(negated)
        if any(neg in word for neg in ["didn't","not", "n't", "no"]):
            negation = True
        else:
            negation=False
    return result

# function used by lemmatizer
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Importing the dataset
# read data
reviews_df = pd.read_csv("hotel-reviews.csv")

# creating corpus of words
corpus=[]
c1=0
c2=0
y=[]
for i in range(len(reviews_df)):
    if(reviews_df["Is_Response"][i]=="happy" and c1<14000):
        review=re.sub('[^a-zA-Z]',' ',reviews_df["Description"][i])
        review=review.lower()
        review=review.split()
        pos_tags = pos_tag(review)
        review = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        review=negate(review)
        review=[(word) for word in review if not word in set(stopwords.words('english'))]
        review=" ".join(review)
        corpus.append(review)
        c1+=1
        y.append(1)
    elif(reviews_df["Is_Response"][i]=="not happy" and c2<14000):
        review=re.sub('[^a-zA-Z]',' ',reviews_df["Description"][i])
        review=review.lower()
        review=review.split()
        pos_tags = pos_tag(review)
        review = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        review=negate(review)
        review=[(word) for word in review if not word in set(stopwords.words('english'))]
        review=" ".join(review)
        corpus.append(review)
        c2+=1
        y.append(0)
    
#creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=10000)
X=cv.fit_transform(corpus).toarray()
lm=cv.vocabulary_



# spilling dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# reducing the number of variables to prevent overfitting
from sklearn.decomposition import PCA
pca=PCA(n_components=3800)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained=pca.explained_variance_ratio_
sum(explained)


# training the model
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2

classifier=Sequential()
classifier.add(Dense(128,input_shape=(3800,),kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(64,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,batch_size=50,epochs=100)
y_pred=classifier.predict(X_test)

y_pred=list(map(int,(y_pred>0.5)))
y_pred=np.reshape(y_pred,(5283,))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# result
'''avg accuracy on training set: 86%
accuracy on test set: 84.74%
precision:85.42%
recall:85.73%'''

# play with the model
review="the cost of food was high"
review=re.sub('[^a-zA-Z]',' ',review)
review=review.lower()
review=review.split()
pos_tags = pos_tag(review)
review = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
review=negate(review)
review=[(word) for word in review if not word in set(stopwords.words('english'))]
review=" ".join(review)
review=cv.transform([review]).toarray()
review=pca.transform(review)
tt=classifier.predict(review)