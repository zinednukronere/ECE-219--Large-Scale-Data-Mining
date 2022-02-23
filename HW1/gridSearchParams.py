#%% Import necessary parameters and set he random seeds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from joblib import Memory
from shutil import rmtree

import random
np.random.seed(42)
random.seed(42)
#%% Load the data
dataframe = pd.read_csv("Project_1_dataset_01_01_2022.csv")
train, test = train_test_split(dataframe[["full_text","root_label"]], test_size=0.2)
#%% Define memory
location = "cachedir"
memory = Memory(location=location, verbose=10)

#%% Define example pipeline
my_pipeline = Pipeline([
    ('clean', Cleaner(toClean=True)),
    ('vectorize',CountVectorizer(min_df=3, 
                             analyzer=lemmaAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=50,random_state = 42)),
    ('model',SVC(kernel = "linear",C = 1000))
],memory=memory)
#%% Define parameters to change
paramGrid = [
     {
         'clean__toClean': [True,False],
          'vectorize__min_df': [3,5],
          'vectorize__analyzer': [lemmaAndRemoveDigits,stemAndRemoveDigits,nothingAndRemoveDigits],
          'dimReduce': [TruncatedSVD(random_state=42),NMF(init='random', random_state=42)],
          'dimReduce__n_components': [500],
          "model":[ SVC(kernel = "linear",C = 100),
                    LogisticRegression(penalty='l1', C=100, solver = "liblinear",max_iter=500),
                    LogisticRegression(penalty='l2', C=1000, solver = "liblinear",max_iter=500),
                     GaussianNB()
                   ]
     }
]
#%% Perform grid search
grid = GridSearchCV(my_pipeline, cv=5, n_jobs=1, param_grid=paramGrid, scoring='accuracy',verbose=10)
grid.fit(train["full_text"], train["root_label"])
results = pd.DataFrame(grid.cv_results_)
results.to_csv("gridResults500GaussianNB.csv", encoding='utf-8', index=False)

#%% Fitting best combination 1
pipe1 = Pipeline([
    ('clean', Cleaner(toClean=False)),
    ('vectorize',CountVectorizer(min_df=5, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=500,random_state = 42)),
    ('model',LogisticRegression(penalty='l2', C=1000, solver = "liblinear",max_iter=500))
],memory=memory)
#%% Evaluating combination
pipe1.fit(train["full_text"], train["root_label"])
acc1,prec1,recall1,f1 = calculatePerfParams(pipe1,test["full_text"],test["root_label"],"climate")
#%% Fitting best combination 2
pipe2 = Pipeline([
    ('clean', Cleaner(toClean=True)),
    ('vectorize',CountVectorizer(min_df=3, 
                             analyzer=stemAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=500,random_state = 42)),
    ('model',LogisticRegression(penalty='l2', C=1000, solver = "liblinear",max_iter=500))
],memory=memory)
#%% Evaluating combination
pipe2.fit(train["full_text"], train["root_label"])
acc2,prec2,recall2,f12 = calculatePerfParams(pipe2,test["full_text"],test["root_label"],"climate")
#%% Fitting best combination 3
pipe3 = Pipeline([
    ('clean', Cleaner(toClean=False)),
    ('vectorize',CountVectorizer(min_df=3, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=500,random_state = 42)),
    ('model',LogisticRegression(penalty='l2', C=1000, solver = "liblinear",max_iter=500))
],memory=memory)
#%% Evaluating combination
pipe3.fit(train["full_text"], train["root_label"])
acc3,prec3,recall3,f13 = calculatePerfParams(pipe3,test["full_text"],test["root_label"],"climate")
#%% Fitting best combination 4
pipe4 = Pipeline([
    ('clean', Cleaner(toClean=True)),
    ('vectorize',CountVectorizer(min_df=5, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=500,random_state = 42)),
    ('model',LogisticRegression(penalty='l2', C=1000, solver = "liblinear",max_iter=500))
],memory=memory)
#%% Evaluating combination
pipe4.fit(train["full_text"], train["root_label"])
acc4,prec4,recall4,f14 = calculatePerfParams(pipe4,test["full_text"],test["root_label"],"climate")
#%% Fitting best combination 5
pipe5 = Pipeline([
    ('clean', Cleaner(toClean=False)),
    ('vectorize',CountVectorizer(min_df=5, 
                             analyzer=lemmaAndRemoveDigits,
                             stop_words='english')),
    ('tfidf',TfidfTransformer()),
    ('dimReduce',TruncatedSVD(n_components=500,random_state = 42)),
    ('model',LogisticRegression(penalty='l1', C=100, solver = "liblinear",max_iter=500))
],memory=memory)
#%% Evaluating combination
pipe5.fit(train["full_text"], train["root_label"])
acc5,prec5,recall5,f15 = calculatePerfParams(pipe5,test["full_text"],test["root_label"],"climate")