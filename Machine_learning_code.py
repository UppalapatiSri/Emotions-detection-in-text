from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import seaborn as sns
import nltk
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

df = pd.read_csv('Emotion_final.csv')
df.shape
df.head()
df.describe()
df.info()

PREPROCESSING DATA
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
vectorizer = CountVectorizer(stop_words='english')
preprocessed_data = []
for text in df['Text']:
    # Tokenize the text
    tokens = nltk.word_tokenize(text) 
    # Remove stopwords, lowercase, and stem
    words = [stemmer.stem(word.lower()) for word in tokens if word.isalpha() and word.lower() not in stop_words]
    # Join the words back into a single string
    preprocessed_data.append(' '.join(words))
    
X = vectorizer.fit_transform(df['Text'])
y = df['Emotion']
# Split the data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

validation_scores,testing_scores,training_score=[],[],[]

MNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_train = mnb.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
training_score.append(accuracy*100)
print("Accuracy on training data:", accuracy*100)
y_test_pred = mnb.predict(X_test)
test_score = accuracy_score(y_test, y_test_pred)
testing_scores.append(test_score*100)
print('Test score:', test_score*100)
y_val_pred = mnb.predict(X_val)
val_score = accuracy_score(y_val, y_val_pred)
validation_scores.append(val_score*100)
print('Validation score:', val_score*100)

SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_train = svm.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
training_score.append(accuracy*100)
print("Accuracy on training data:", accuracy*100)
y_test_pred = svm.predict(X_test)
test_score = accuracy_score(y_test, y_test_pred)
testing_scores.append(test_score*100)
print('Test score:', test_score*100)
y_val_pred = svm.predict(X_val)
val_score = accuracy_score(y_val, y_val_pred)
validation_scores.append(val_score*100)
print('Validation score:', val_score*100)

RD
rd = RandomForestClassifier(n_estimators=100,random_state=42)
rd.fit(X_train, y_train)
y_pred_train = rd.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
training_score.append(accuracy*100)
print("Accuracy on training data:", accuracy*100)
y_test_pred = rd.predict(X_test)
test_score = accuracy_score(y_test, y_test_pred)
testing_scores.append(test_score*100)
print('Test score:', test_score*100)
y_val_pred = rd.predict(X_val)
val_score = accuracy_score(y_val, y_val_pred)
validation_scores.append(val_score*100)
print('Validation score:', val_score*100)

models=[' Multinomial Naive Bayes','Support Vector Machines','Random Forest']
from tabulate import tabulate
data = []
for i in range(len(models)):
    data.append([models[i],training_score[i],testing_scores[i], validation_scores[i]])

# Print the table
print(tabulate(data, headers=['Model','Train_Score', 'Test_Score', 'Validation_Score']))

preprocessed_text = vectorizer.transform(["i have horrible anxiety dreams"])
predicted_label1 = mnb.predict(preprocessed_text)[0]
predicted_label2 = svm.predict(preprocessed_text)[0]
predicted_label3 = rd.predict(preprocessed_text)[0]
from collections import Counter
b=[predicted_label1,predicted_label2,predicted_label3]
d=Counter(b)
n=[i for i,j in d.items() if j==max(d.values())]
h = {'sadness':"😞", 'surprise':"😮",'happy':"😄",'angry':"😡",'fear':"😱"}
print(n[0],h[n[0]])
