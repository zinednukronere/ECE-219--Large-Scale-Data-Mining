#Loading the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms.matrix_factorization import NMF

#%% Loadingthe dataset
ratingData = pd.read_csv("ratings.csv")
userIds = ratingData['userId']
movieIds = ratingData['movieId']
ratings = ratingData['rating']
Rmatrix = ratingData.pivot_table('rating', 'userId', 'movieId', fill_value=0)

reader = Reader(rating_scale=(0.5, 5))
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)

#%% QUESTION 8: Designing the NMF Collaborative Filter:
avg_rmse = []
avg_mae = []
ks = np.linspace(2,50,num=25,dtype=int)
for k in ks:
    print(k)
    perf = cross_validate(NMF(n_factors=k,verbose=False),ratingsSurpriseSet,cv=10)
    avg_rmse.append(np.mean(perf['test_rmse']))
    avg_mae.append(np.mean(perf['test_mae']))

print("Minimum average RMSE: ", min(avg_rmse))
print("Minimum average MAE: ", min(avg_mae))
#%% Plot NMF Error
ks = np.linspace(2,50,num=25,dtype=int)
plt.plot(ks,avg_rmse, label='Average RMSE')
plt.plot(ks, avg_mae, label='Average MAE')
plt.scatter(ks[np.argmin(avg_rmse)],min(avg_rmse),marker='o',label='min RMSE')
plt.scatter(ks[np.argmin(avg_mae)],min(avg_mae),marker='o',label='min MAE')
plt.xlabel("k"); plt.ylabel("Error"); plt.title("NMF collaborative filter with 10-fold CV")
plt.legend()

#%% Number of movie genres
movies = pd.read_csv('movies.csv',names=['movieid','title','genres'],header=0)
genres = []

for movie_genre in movies.genres:
    genres += movie_genre.split('|')
print(len(set(genres)))

#%% Helper functions for test set subsets
def filterPopular(testSet,Rmatrix):
    movieFreq = np.sum(Rmatrix!=0, axis=0)
    freqMovIds = list(movieFreq.where(movieFreq >2).dropna().keys())
    passedItems=[]
    for item in testSet:
        movId = item[1]
        if movId in freqMovIds :
            passedItems.append(item)
    return passedItems 

def filterUnpopular(testSet,Rmatrix):
    movieFreq = np.sum(Rmatrix!=0, axis=0)
    freqMovIds = list(movieFreq.where(movieFreq <= 2).dropna().keys())
    passedItems=[]
    for item in testSet:
        movId = item[1]
        if movId in freqMovIds :
            passedItems.append(item)
    return passedItems 

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

#%% Popular Movies
from surprise.model_selection import KFold
from surprise import accuracy

kfold = KFold(n_splits=10)
rmsePopMov= []
for k in ks:
    nmf = NMF(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = filterPopular(testset,Rmatrix)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmsePopMov.append(np.mean(foldrsmes))
#%% plot Popular Movies CV
plt.figure()
plt.plot(ks,rmsePopMov)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("NMF collaborative filter with 10-fold CV on popular movies") 
print("Minimum average RMSE: ", min(rmsePopMov))


#%% Unpopular Movies

kfold = KFold(n_splits=10)
rmseUnPopMov= []
for k in ks:
    nmf = NMF(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = filterUnpopular(testset, Rmatrix)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmseUnPopMov.append(np.mean(foldrsmes))
#%% plot Unpopular Movies CV
plt.figure()
plt.plot(ks,rmseUnPopMov)
plt.xlabel("k")
plt.ylabel("Average RMSE")   
plt.title("NMF collaborative filter with 10-fold CV on unpopular movies") 
print("Minimum average RMSE: ", min(rmseUnPopMov))

#%% High Variance Movies
Rmatrix_na = ratingData.pivot_table('rating', 'userId', 'movieId') # no rating as nan
variances = np.var(Rmatrix_na, axis=0)

kfold = KFold(n_splits=10)
rmseVarMov= []
for k in ks:
    nmf = NMF(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = highVarianceCheck(testset,Rmatrix,variances)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmseVarMov.append(np.mean(foldrsmes))
    
#%% plot High Variance Movies CV
plt.figure()
plt.plot(ks,rmseVarMov)
plt.xlabel("k")
plt.ylabel("Average RMSE")   
plt.title("NMF collaborative filter with 10-fold CV on high variance movies") 
print("Minimum average RMSE: ", min(rmseVarMov))
#%% ROC curves for the NMF
from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_auc_score
from surprise.model_selection import train_test_split

k = 20
ratingThresholds = [2.5,3,3.5,4]

fprs = []
tprs = []
AUCs = []

for threshold in ratingThresholds:
    nmf = NMF(n_factors=k,verbose=False)
    trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
    nmf.fit(trainset)
    pred = nmf.test(testset)

    real_y = []
    est_y = []
    for i in range(len(pred)):
        est_y.append(pred[i].est)
        if testset[i][2] >= threshold: #ratings threshold
            real_y.append(1.0)
        else:
            real_y.append(0.0)
            
    fpr, tpr, thresholds = roc_curve(real_y, est_y)
    AUC = roc_auc_score(real_y, est_y)
    fprs.append(fpr)
    tprs.append(tpr)
    AUCs.append(AUC)


#%% 
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=23)
for i in range(len(fprs)):
    plt.plot(fprs[i],tprs[i], label = 'ROC Curve: \
             Threshold: {rt} AUC: {auc}'.format(rt=ratingThresholds[i], auc=AUCs[i]))
plt.legend(loc="lower right")
plt.show()

#%% QUESTION 9: Interpreting the NMF model
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)

trainset, testset = train_test_split(ratingsSurpriseSet, test_size=.1)
nmf = NMF(n_factors=20,verbose=False)
nmf.fit(trainset).test(testset)
V = nmf.qi
cols = [0,5,10,15,19]
df = pd.read_csv('./Synthetic_Movie_Lens/movies.csv',names=['movieid','title','genres'],header=0)
for i in cols:
    print('\nlf:' + str(i+1))
    col = V[:,i]
    movies = [(n,j) for n,j in enumerate(col)]
    
    movies.sort(key = lambda x:x[1], reverse=True)
    
    for movie in movies[:10]:
        print(df['genres'][movie[0]])
      
#%% QUESTION 10: Designing the MF Collaborative Filter:
from surprise.prediction_algorithms.matrix_factorization import SVD

avg_rmse = []
avg_mae = []
ks = np.linspace(2,50,num=25,dtype=int)
for k in ks:
    print(k)
    perf = cross_validate(SVD(n_factors=k,verbose=False),ratingsSurpriseSet,cv=10)
    avg_rmse.append(np.mean(perf['test_rmse']))
    avg_mae.append(np.mean(perf['test_mae']))

print("Minimum average RMSE: ", min(avg_rmse))
print("Minimum average MAE: ", min(avg_mae))
#%% Plot MF
plt.plot(ks,avg_rmse,  label='Average RMSE')
plt.plot(ks, avg_mae,  label='Average MAE')

plt.scatter(ks[np.argmin(avg_rmse)],min(avg_rmse),marker='o',label='min RMSE')
plt.scatter(ks[np.argmin(avg_mae)],min(avg_mae),marker='o',label='min MAE')

plt.legend(loc='best')
plt.xlabel("k")
plt.ylabel("Error")
plt.title("MF 10-fold CV")
plt.show()

#%% Popular Movies

ks = np.linspace(2,50,num=25,dtype=int)
kfold = KFold(n_splits=10)
rmsePopMov= []
for k in ks:
    nmf = SVD(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = filterPopular(testset,Rmatrix)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmsePopMov.append(np.mean(foldrsmes))
#%% plot Popular Movies CV
plt.figure()
plt.plot(ks,rmsePopMov)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("MF 10-fold CV on popular movies") 
print("Minimum average RMSE: ", min(rmsePopMov))

#%% Unpopular Movies

ks = np.linspace(2,50,num=25,dtype=int)
kfold = KFold(n_splits=10)
rmseUnPopMov= []
for k in ks:
    nmf = SVD(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = filterUnpopular(testset,Rmatrix)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmseUnPopMov.append(np.mean(foldrsmes))
#%% plot Unpopular Movies CV
plt.figure()
plt.plot(ks,rmseUnPopMov)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("MF 10-fold CV on unpopular movies") 
print("Minimum average RMSE: ", min(rmseUnPopMov))
#%% HighVar Movies

ks = np.linspace(2,50,num=25,dtype=int)
kfold = KFold(n_splits=10)
rmsehighvarpMov= []
for k in ks:
    nmf = SVD(n_factors=k,verbose=False)
    foldrsmes = []
    print(k)
    for trainset, testset in kfold.split(ratingsSurpriseSet):
        #train and test
        nmf.fit(trainset)
        filteredTest = highVarianceCheck(testset, Rmatrix, variances)
        preds = nmf.test(filteredTest)
        #compute RMSE
        foldrsmes.append(accuracy.rmse(preds, verbose=False))
    rmsehighvarpMov.append(np.mean(foldrsmes))
#%% plot HighVar Movies CV
plt.figure()
plt.plot(ks,rmsehighvarpMov)
plt.xlabel("k")
plt.ylabel("rmse")   
plt.title("MF 10-fold CV on high var movies") 
print("Minimum average RMSE: ", min(rmsehighvarpMov))

#%% ROC curves for the MF

k = 24
ratingThresholds = [2.5,3,3.5,4]

fprs = []
tprs = []
AUCs = []

for threshold in ratingThresholds:
    svd = SVD(n_factors=k,verbose=False)
    trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
    svd.fit(trainset)
    pred = svd.test(testset)

    real_y = []
    est_y = []
    for i in range(len(pred)):
        est_y.append(pred[i].est)
        if testset[i][2] >= threshold: #ratings threshold
            real_y.append(1.0)
        else:
            real_y.append(0.0)
            
    fpr, tpr, thresholds = roc_curve(real_y, est_y)
    AUC = roc_auc_score(real_y, est_y)
    fprs.append(fpr)
    tprs.append(tpr)
    AUCs.append(AUC)
    
#%% Plot ROC Curve
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=23)
for i in range(len(fprs)):
    plt.plot(fprs[i],tprs[i], label = 'ROC Curve: Threshold: {rt} AUC: {auc}'.format(rt=ratingThresholds[i], auc=AUCs[i]))
plt.legend(loc="lower right")
plt.show()