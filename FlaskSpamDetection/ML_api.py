import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.metrics import classification_report
from sklearn import feature_extraction
#from sklearn.metrics import roc_auc_score
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings('ignore')

import dill as pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier

{'clf__alpha': 0.01,
 'tfidf__use_idf': True,
 'vect__analyzer': 'char',
 'vect__ngram_range': (2, 3)}


def build_and_train():
    path='/Users/KBrig/ML-Pipeline/spam.csv'
    EmData= pd.read_csv('Spam.csv',encoding='cp1252',dtype={'type':np.str, 'Email_body':np.str})
    EmData=EmData[['type','Email_body']]
    EmData['type_num']= EmData.type.map({'ham':0, 'spam':1})

    X= EmData.Email_body
    y= EmData.type_num

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
    def pip(classifier):
        pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=(2,3), analyzer='char')),  # strings to token integer counts
            ('tfidf', TfidfTransformer(use_idf=True)),  # integer counts to weighted TF-IDF scores
            ('clf', classifier),
        ])
        return(pipeline)
    pip_SVM = pip(SGDClassifier(alpha=0.01))
    pip_SVM.fit(X_train,y_train)
    return(pip_SVM)

if __name__ == '__main__':
    model = build_and_train()
    filename = 'model_SVM.pk'
    with open('C:/Users/KBrig/Desktop/FlaskSpamDetection/models/'+filename, 'wb') as file:
        pickle.dump(model, file)
