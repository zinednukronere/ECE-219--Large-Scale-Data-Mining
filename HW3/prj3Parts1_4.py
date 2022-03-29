#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Loading the data
ratingData = pd.read_csv("ratings.csv")
userIds = ratingData['userId']
movieIds = ratingData['movieId']
ratings = ratingData['rating']
#%% Creating the R matrix
Rmatrix = ratingData.pivot_table('rating', 'userId', 'movieId', fill_value=0)
#%%Calculating sparsity
totalPossible = Rmatrix.shape[0] * Rmatrix.shape[1]
ratingAmount = len(ratings)
sparsity = ratingAmount/totalPossible
#%% Plotting the histogram of the ratings
bins = np.linspace(0,5,num=11)
plt.hist(ratings,bins=bins)
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("Histogram of rating values")
#%% Plotting the number of ratings among movies
movieFrequency = sorted(list(np.sum(Rmatrix!=0, axis=0)),reverse=True)
x = np.arange(0,len(movieFrequency))
plt.figure()
plt.plot(x,movieFrequency)
plt.xlabel("Movie index")
plt.ylabel("Rating Frequency")
plt.title("Number of ratings among movies")
plt.show()
#%%Plotting the number of ratings among users
userFrequency = sorted(list(np.sum(Rmatrix!=0, axis=1)),reverse=True)
x = np.arange(0,len(userFrequency))
plt.figure()
plt.plot(x,userFrequency)
plt.xlabel("User index")
plt.ylabel("Rating Frequency")
plt.title("Number of ratings among users")
plt.show()
#%% Plotting the histogram of variances
Rmatrix_na = ratingData.pivot_table('rating', 'userId', 'movieId') # no rating as nan
variances = np.var(Rmatrix_na, axis=0)
bins=np.arange(min(variances),max(variances)+0.5,0.5)
plt.figure()
plt.hist(variances,bins=bins)
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.title("Histogram of variances")
#%% Sweeping k values for knn 
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.knns import KNNWithMeans

reader = Reader(rating_scale=(0.5, 5))
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)

rmse = []
mae = []

for k in range(2,102,2):
    knn = KNNWithMeans(k=k, sim_options = {'name': 'pearson'})
    cv = cross_validate(knn, ratingsSurpriseSet, measures=['RMSE','MAE'],cv=10, verbose=True)
    rmse.append(np.mean(cv['test_rmse']))
    mae.append(np.mean(cv['test_mae']))
#%% Saving results so far for easier access
dictionaryKNN = {"knnrmse":rmse,"knnmae":mae}
import pickle

with open('dictionaryKNN.pickle', 'wb') as handle:
    pickle.dump(dictionaryKNN,handle)
                
#%% Loading the results
import pickle
file = open("dictionaryKNN.pickle","rb")
dictionaryKNN = pickle.load(file)
rmses = dictionaryKNN['knnrmse']
maes = dictionaryKNN['knnmae']
#%% Plotting mae and rmse vs k
x = np.arange(2,102,2)
plt.figure()
plt.plot(x,rmses)
plt.xlabel("k")
plt.ylabel("rmse")
plt.title("Average RMSE vs k")

plt.figure()
plt.plot(x,maes)
plt.xlabel("k")
plt.ylabel("mae")
plt.title("Average MAE vs k")

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
    
#%% For Popular Movies
from surprise.model_selection import KFold
from surprise import accuracy

kfold = KFold(n_splits=10)
rmsePopMov= []
for k in range(2,102,2):
    knn = KNNWithMeans(k=k, sim_options = {'name': 'pearson'})
    foldrsmes = []
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        knn.fit(trainset)
        filteredTest = filterPopular(testset,Rmatrix)
        preds = knn.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmsePopMov.append(np.mean(foldrsmes))
    
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

#%% For unpopular movies
kfold = KFold(n_splits=10)
rmseUnPopMov= []
for k in range(2,102,2):
    knn = KNNWithMeans(k=k, sim_options = {'name': 'pearson'})
    foldrsmes = []
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        knn.fit(trainset)
        filteredTest = filterUnpopular(testset,Rmatrix)
        preds = knn.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmseUnPopMov.append(np.mean(foldrsmes))
    
#%% Saving
dictionarypopularUnpopular = {"poprmse":rmsePopMov,"unpoprmse":rmseUnPopMov}
import pickle

with open('dictionarypopularUnpopular.pickle', 'wb') as handle:
    pickle.dump(dictionarypopularUnpopular,handle)

#%% 
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
                
#%% KNN with High Variance Check
kfold = KFold(n_splits=10)
rmseVarMov= []
for k in range(2,102,2):
    knn = KNNWithMeans(k=k, sim_options = {'name': 'pearson'})
    foldrsmes = []
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        knn.fit(trainset)
        filteredTest = highVarianceCheck(testset,Rmatrix,variances)
        preds = knn.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmseVarMov.append(np.mean(foldrsmes))

#%% Saving
dictionarypopularAndVariance = {"varrmse":rmseVarMov}
import pickle

with open('dictionaryVariance.pickle', 'wb') as handle:
    pickle.dump(dictionarypopularAndVariance,handle)
    
#%% Loading results for popular and unpopular
import pickle
file = open("dictionarypopularUnpopular.pickle","rb")
dictionarypopularUnpopular = pickle.load(file)
popularRMSE = dictionarypopularUnpopular['poprmse']
unpopularRMSE = dictionarypopularUnpopular['unpoprmse']
#%% Plotting the RMSE curve for popular
x = np.arange(2,102,2)
plt.figure()
plt.plot(x,popularRMSE)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("Rmse vs k for popular movies") 
#%% 
minRMSE = np.min(popularRMSE)
minK = np.argmin(popularRMSE)*2+2
#Best rmse obtained when k=54
#%% Plot ROC curve for popular
from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_auc_score
from surprise.model_selection import train_test_split

ratingThresholds = [2.5,3,3.5,4]

fprsPop = []
tprsPop = []
AUCsPop = []

for threshold in ratingThresholds:
    knn = KNNWithMeans(k=54, sim_options={'name': 'pearson'})
    trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
    knn.fit(trainset)
    filteredTest = filterPopular(testset,Rmatrix)
    pred = knn.test(filteredTest)

    real_y = []
    est_y = []
    for i in range(len(pred)):
        est_y.append(pred[i].est)
        if filteredTest[i][2] >= threshold: #ratings threshold
            real_y.append(1.0)
        else:
            real_y.append(0.0)
            
    fpr, tpr, thresholds = roc_curve(real_y, est_y)
    AUC = roc_auc_score(real_y, est_y)
    fprsPop.append(fpr)
    tprsPop.append(tpr)
    AUCsPop.append(AUC)

#%% 
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve for Popular Trim', fontsize=23)
for i in range(len(fprsPop)):
    plt.plot(fprsPop[i],tprsPop[i], label = 'ROC Curve: Threshold: {rt} AUC: {auc}'.format(rt=ratingThresholds[i], auc=AUCsPop[i]))
plt.legend(loc="lower right")
plt.show()

#%% Plotting the RMSE for unpopular
x = np.arange(2,102,2)
plt.figure()
plt.plot(x,unpopularRMSE)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("Rmse vs k for unpopular movies") 
#%% 
minRMSE = np.min(unpopularRMSE)
minK = np.argmin(unpopularRMSE)*2+2
#Best rmse obtained when k=16
#%% ROC for unpopular
ratingThresholds = [2.5,3,3.5,4]

fprsUnpop = []
tprsUnpop = []
AUCsUnpop = []

for threshold in ratingThresholds:
    knn = KNNWithMeans(k=16, sim_options={'name': 'pearson'})
    trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
    knn.fit(trainset)
    filteredTest = filterUnpopular(testset,Rmatrix)
    pred = knn.test(filteredTest)

    real_y = []
    est_y = []
    for i in range(len(pred)):
        est_y.append(pred[i].est)
        if filteredTest[i][2] >= threshold: #ratings threshold
            real_y.append(1.0)
        else:
            real_y.append(0.0)
            
    fpr, tpr, thresholds = roc_curve(real_y, est_y)
    AUC = roc_auc_score(real_y, est_y)
    fprsUnpop.append(fpr)
    tprsUnpop.append(tpr)
    AUCsUnpop.append(AUC)

#%% 
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve for Unpopular Trim', fontsize=23)
for i in range(len(fprsUnpop)):
    plt.plot(fprsUnpop[i],tprsUnpop[i], label = 'ROC Curve: Threshold: {rt} AUC: {auc}'.format(rt=ratingThresholds[i], auc=AUCsUnpop[i]))
plt.legend(loc="lower right")
plt.show()

#%% Loading results for high variance
import pickle
file = open("dictionaryVariance.pickle","rb")
dictionaryVar = pickle.load(file)
varRMSE = dictionaryVar['varrmse']

#%% Plotting the RMSE for high variance
x = np.arange(2,102,2)
plt.figure()
plt.plot(x,varRMSE)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("Rmse vs k for high variance movies") 
#%%
minRMSE = np.min(varRMSE)
minK = np.argmin(varRMSE)*2+2
#Best rmse obtained when k=16
#%% ROC for high variance
ratingThresholds = [2.5,3,3.5,4]

fprsHighVar = []
tprsHighVar = []
AUCsHighVar = []

for threshold in ratingThresholds:
    knn = KNNWithMeans(k=88, sim_options={'name': 'pearson'})
    trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
    knn.fit(trainset)
    filteredTest = highVarianceCheck(testset,Rmatrix,variances)
    pred = knn.test(filteredTest)

    real_y = []
    est_y = []
    for i in range(len(pred)):
        est_y.append(pred[i].est)
        if filteredTest[i][2] >= threshold: #ratings threshold
            real_y.append(1.0)
        else:
            real_y.append(0.0)
            
    fpr, tpr, thresholds = roc_curve(real_y, est_y)
    AUC = roc_auc_score(real_y, est_y)
    fprsHighVar.append(fpr)
    tprsHighVar.append(tpr)
    AUCsHighVar.append(AUC)
    
#%% 
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve for High Var Trim', fontsize=23)
for i in range(len(fprsHighVar)):
    plt.plot(fprsHighVar[i],tprsHighVar[i], label = 'ROC Curve: Threshold: {rt} AUC: {auc}'.format(rt=ratingThresholds[i], auc=AUCsHighVar[i]))
plt.legend(loc="lower right")
plt.show()




