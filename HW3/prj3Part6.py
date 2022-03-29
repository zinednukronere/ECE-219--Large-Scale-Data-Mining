#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise.model_selection import KFold
from surprise import Dataset, Reader
from sklearn.metrics import mean_squared_error

#%% Loading the data
ratingData = pd.read_csv("ratings.csv")
userIds = ratingData['userId']
movieIds = ratingData['movieId']
ratings = ratingData['rating']
#%% Preparing the surprise set
reader = Reader(rating_scale=(0.5, 5))
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)
#%% Creating the R matrix
Rmatrix_na = ratingData.pivot_table('rating', 'userId', 'movieId')
userMeans = np.mean(Rmatrix_na,axis=1)
Rmatrix = ratingData.pivot_table('rating', 'userId', 'movieId', fill_value=0)
#%% Performing Kfold cross validation
kFoldrmses = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(ratingsSurpriseSet):
    pred = [userMeans[i[0]] for i in testset]
    true = [i[2] for i in testset]
    kFoldrmses.append(np.sqrt(mean_squared_error(true,pred)))
avgRmse = np.mean(kFoldrmses)
#%%
def filterPopular(testSet,Rmatrix):
    movieFreq = np.sum(Rmatrix!=0, axis=0)
    freqMovIds = list(movieFreq.where(movieFreq >2).dropna().keys())
    passedItems=[]
    for item in testSet:
        movId = item[1]
        if movId in freqMovIds :
            passedItems.append(item)
    return passedItems 
    
#%% cross validation for popular
kfold = KFold(n_splits=10)
kFoldrmsesPopular= []
for trainset, testset in kfold.split(ratingsSurpriseSet):
    filteredTest = filterPopular(testset,Rmatrix)
    pred = [userMeans[i[0]] for i in filteredTest]
    true = [i[2] for i in filteredTest]
    kFoldrmsesPopular.append(np.sqrt(mean_squared_error(true,pred)))
avgRmsesPopular = np.mean(kFoldrmsesPopular)

#%% 
def filterUnpopular(testSet,Rmatrix):
    movieFreq = np.sum(Rmatrix!=0, axis=0)
    freqMovIds = list(movieFreq.where(movieFreq <= 2).dropna().keys())
    passedItems=[]
    for item in testSet:
        movId = item[1]
        if movId in freqMovIds :
            passedItems.append(item)
    return passedItems 
#%% Cross validation for unpopular
kfold = KFold(n_splits=10)
kFoldrmsesUnpopular= []
for trainset, testset in kfold.split(ratingsSurpriseSet):
    filteredTest = filterUnpopular(testset,Rmatrix)
    pred = [userMeans[i[0]] for i in filteredTest]
    true = [i[2] for i in filteredTest]
    kFoldrmsesUnpopular.append(np.sqrt(mean_squared_error(true,pred)))
avgRmsesUnpopular = np.mean(kFoldrmsesUnpopular)
#%%
variances = np.var(Rmatrix_na, axis=0)

def highVarianceCheck(testSet,Rmatrix,variances):
    movieFreq = np.sum(Rmatrix!=0, axis=0)
    freqMovIds = list(movieFreq.where(movieFreq >=5).dropna().keys())
    highVarMovIds = list(variances.where(variances >=2).dropna().keys())    
    passedItems=[]
    for item in testSet:
        movId = item[1]
        if movId in freqMovIds and movId in highVarMovIds:
            passedItems.append(item)
    return passedItems

#%% Cross validation high variance
kfold = KFold(n_splits=10)
kFoldrmsesHighVar= []
for trainset, testset in kfold.split(ratingsSurpriseSet):
    filteredTest = highVarianceCheck(testset,Rmatrix,variances)
    pred = [userMeans[i[0]] for i in filteredTest]
    true = [i[2] for i in filteredTest]
    kFoldrmsesHighVar.append(np.sqrt(mean_squared_error(true,pred)))
avgRmsesHighVar = np.mean(kFoldrmsesHighVar)
