#%% Import the libraries
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise.model_selection import KFold
from surprise import Dataset, Reader
from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_auc_score
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNWithMeans
#%% Loadind the data
ratingData = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(0.5, 5))
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)
#%% Getting rid of the users that have rated less than t and didnt like any movies 
def filterUsers(testSet,t,threshold):
    timesUserRated = {}
    timesUserLiked = {}
    
    for (userId, movieId, rating) in testSet:
        if userId not in timesUserRated:
            timesUserRated[userId] = 0
        timesUserRated[userId] += 1
        if userId not in timesUserLiked:
            timesUserLiked[userId] = 0
        if rating >= threshold:
            timesUserLiked[userId] += 1
            
    remaining = []
    for (userId, movieId, rating) in testSet:
        if timesUserRated[userId] >= t and timesUserLiked[userId] > 0:
            remaining.append((userId, movieId, rating))
    
    return remaining
#%% Calculate the precision and recall across users and get their means
def calculatePrecisionRecall(predictions, t, threshold=3):

    userResults = {}
    for userid, _, realrate, predrate, _ in predictions:
        if userid not in userResults:
            userResults[userid]=[]
        userResults[userid].append((realrate, predrate))
    
    precisions = {}
    recalls = {}
    for userid, results in userResults.items():
        
        # Sort by estimated value
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Number of relevant and recommended items in top t
        noRelandRec = 0
        for i in range(t):
            if results[i][0] >= threshold:
                noRelandRec += 1
        
        # Number of relevant items
        noRel = sum((real >= threshold) for (real,pred) in results)
    
        # Precision
        precisions[userid] = noRelandRec / t
    
        # Recall
        recalls[userid] = noRelandRec / noRel
        
    meanPrecisionAcrossUsers = sum([prec for prec in precisions.values()]) / len(precisions)
    meanRecallAcrossUsers = sum([recall for recall in recalls.values()]) / len(recalls)
    
    return meanPrecisionAcrossUsers, meanRecallAcrossUsers

#%% Generate precision and recall means for knn for different ts
threshold = 3
ts = [i for i in range(1, 25 + 1)]
bestK = 20

knnTPrecisions = []
knnTRecalls = []
kf = KFold(n_splits=10)

for t in ts:
    foldPrecs = []
    foldRecs = []

    for trainset, testset in kf.split(ratingsSurpriseSet):
        knn = KNNWithMeans(k=bestK, sim_options={'name': 'pearson'})
        knn.fit(trainset)
        filteredTest = filterUsers(testset, t, threshold)
        preds = knn.test(filteredTest)
    
        meanUserPrecisions, meanUserRecalls = calculatePrecisionRecall(preds, t, threshold=3)
            
        foldPrecs.append(meanUserPrecisions)
        foldRecs.append(meanUserRecalls)
    
    meanFoldPrecs = np.mean(foldPrecs)
    meanFoldRecs = np.mean(foldRecs)
    knnTPrecisions.append(meanFoldPrecs)
    knnTRecalls.append(meanFoldRecs)    
    
#%% Saving results so far 
dictionaryKNNPrecsAndRecs = {"knnprecisions":knnTPrecisions,"knnrecalls":knnTRecalls}
import pickle

with open('dictionaryKNNPrecsAndRecs.pickle', 'wb') as handle:
    pickle.dump(dictionaryKNNPrecsAndRecs,handle)
    
#%% Loading the results
import pickle
file = open("dictionaryKNNPrecsAndRecs.pickle","rb")
dictionaryKNNPrecsAndRecs = pickle.load(file)
knnTPrecisions = dictionaryKNNPrecsAndRecs['knnprecisions']
knnTRecalls = dictionaryKNNPrecsAndRecs['knnrecalls']
#%% Function to scatter
def generatePlots(scatterX,scatterY,xLabel,yLabel,title):
    plt.figure()
    plt.scatter(scatterX,scatterY,s=30)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()
#%% Plotting for knn
ts = [i for i in range(1, 25 + 1)]

generatePlots(ts,knnTPrecisions,"t","Precision","Precision vs t for knn")
generatePlots(ts,knnTRecalls,"t","Recall","Recall vs t for knn")
generatePlots(knnTRecalls,knnTPrecisions,"Recall","Precision","Precision vs Recall for knn")

#%% Getting the precision and recall vales for NMF for different ts
from surprise.prediction_algorithms.matrix_factorization import NMF

threshold = 3
ts = [i for i in range(1, 25 + 1)]
bestNMF = 18

NMFTPrecisions = []
NMFTRecalls = []
kf = KFold(n_splits=10)

for t in ts:
    foldPrecs = []
    foldRecs = []

    for trainset, testset in kf.split(ratingsSurpriseSet):
        nmf = NMF(n_factors=bestNMF,verbose=False)
        nmf.fit(trainset)
        filteredTest = filterUsers(testset, t, threshold)
        preds = nmf.test(filteredTest)
    
        meanUserPrecisions, meanUserRecalls = calculatePrecisionRecall(preds, t, threshold=3)
            
        foldPrecs.append(meanUserPrecisions)
        foldRecs.append(meanUserRecalls)
    
    meanFoldPrecs = np.mean(foldPrecs)
    meanFoldRecs = np.mean(foldRecs)
    NMFTPrecisions.append(meanFoldPrecs)
    NMFTRecalls.append(meanFoldRecs)      

    
#%% Saving results so far 
dictionaryNMFPrecsAndRecs = {"nmfprecisions":NMFTPrecisions,"nmfrecalls":NMFTRecalls}
import pickle

with open('dictionaryNMFPrecsAndRecs.pickle', 'wb') as handle:
    pickle.dump(dictionaryNMFPrecsAndRecs,handle)    
#%% Loading the results
import pickle
file = open("dictionaryNMFPrecsAndRecs.pickle","rb")
dictionaryNMFPrecsAndRecs = pickle.load(file)
NMFTPrecisions = dictionaryNMFPrecsAndRecs['nmfprecisions']
NMFTRecalls = dictionaryNMFPrecsAndRecs['nmfrecalls']
#%% Plot for NMF
ts = [i for i in range(1, 25 + 1)]

generatePlots(ts,NMFTPrecisions,"t","Precision","Precision vs t for NMF")
generatePlots(ts,NMFTRecalls,"t","Recall","Recall vs t for NMF")
generatePlots(NMFTRecalls,NMFTPrecisions,"Recall","Precision","Precision vs Recall for NMF")


#%% Getting the precision and recall vales for MF with bias for different ts

from surprise.prediction_algorithms.matrix_factorization import SVD

threshold = 3
ts = [i for i in range(1, 25 + 1)]
bestSVD = 24

SVDTPrecisions = []
SVDTRecalls = []
kf = KFold(n_splits=10)

for t in ts:
    foldPrecs = []
    foldRecs = []

    for trainset, testset in kf.split(ratingsSurpriseSet):
        svd = SVD(n_factors=bestSVD,verbose=False)
        svd.fit(trainset)
        filteredTest = filterUsers(testset, t, threshold)
        preds = svd.test(filteredTest)
    
        meanUserPrecisions, meanUserRecalls = calculatePrecisionRecall(preds, t, threshold=3)
            
        foldPrecs.append(meanUserPrecisions)
        foldRecs.append(meanUserRecalls)
    
    meanFoldPrecs = np.mean(foldPrecs)
    meanFoldRecs = np.mean(foldRecs)
    SVDTPrecisions.append(meanFoldPrecs)
    SVDTRecalls.append(meanFoldRecs)      
    
#%% Saving results so far 
dictionarySVDPrecsAndRecs = {"svdprecisions":SVDTPrecisions,"svdrecalls":SVDTRecalls}
import pickle

with open('dictionarySVDPrecsAndRecs.pickle', 'wb') as handle:
    pickle.dump(dictionarySVDPrecsAndRecs,handle)    
#%% Loading the results
import pickle
file = open("dictionarySVDPrecsAndRecs.pickle","rb")
dictionarySVDPrecsAndRecs = pickle.load(file)
SVDTPrecisions = dictionarySVDPrecsAndRecs['svdprecisions']
SVDTRecalls = dictionarySVDPrecsAndRecs['svdrecalls']
#%% Plotting the MF with bias
ts = [i for i in range(1, 25 + 1)]

generatePlots(ts,SVDTPrecisions,"t","Precision","Precision vs t for MF with bias")
generatePlots(ts,SVDTRecalls,"t","Recall","Recall vs t for MF with bias")
generatePlots(SVDTRecalls,SVDTPrecisions,"Recall","Precision","Precision vs Recall for MF with bias")

#%% Plotting the preciison vs recall graphs for the three models
plt.figure()
plt.scatter(knnTRecalls,knnTPrecisions,s=30,label="k-NN")
plt.scatter(NMFTRecalls,NMFTPrecisions,s=30,label="NMF")
plt.scatter(SVDTRecalls,SVDTPrecisions,s=30,label="MF with bias")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall for different models")
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
