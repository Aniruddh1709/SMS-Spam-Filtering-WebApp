import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
df=df.rename(columns={'v2':'message'})
X = df['message']
y = df['label']
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
import pickle
vec_file = 'vectorizer.pickle'
pickle.dump(cv, open(vec_file, 'wb'))

from sklearn.externals import joblib
joblib.dump(clf, 'NB_spam_model.pkl')

NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)