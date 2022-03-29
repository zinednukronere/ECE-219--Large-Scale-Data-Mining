#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader
from sklearn.metrics import  roc_curve
from sklearn.metrics import roc_auc_score
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNWithMeans

#%% Loading the data
ratingData = pd.read_csv("ratings.csv")
reader = Reader(rating_scale=(0.5, 5))
ratingsSurpriseSet = Dataset.load_from_df(ratingData[['userId','movieId','rating']], reader)
#%% 
threshold = 3

fprsDiffModels = []
tprsDiffModels = []
AUCsDiffModels = []

trainset, testset = train_test_split(ratingsSurpriseSet, test_size=0.1)
#%% Calculate fprs and tprs for knn
#Best knn on all data previously found to be k=20 using cross validation
knn = KNNWithMeans(k=20, sim_options={'name': 'pearson'})
knn.fit(trainset)
knnPred = knn.test(testset)

    
real_y = []
est_y = []
for i in range(len(knnPred)):
    est_y.append(knnPred[i].est)
    if testset[i][2] >= threshold: 
        real_y.append(1.0)
    else:
        real_y.append(0.0)
            
fpr, tpr, thresholds = roc_curve(real_y, est_y)
AUC = roc_auc_score(real_y, est_y)
fprsDiffModels.append(fpr)
tprsDiffModels.append(tpr)
AUCsDiffModels.append(AUC)

#%% Calculate fprs and tprs for NMF
from surprise.prediction_algorithms.matrix_factorization import NMF

#Best NMF on all data previously found to be k=18 using cross validation
nmf = NMF(n_factors=18,verbose=False)
nmf.fit(trainset)
nmfPred = nmf.test(testset)

    
real_y = []
est_y = []
for i in range(len(nmfPred)):
    est_y.append(nmfPred[i].est)
    if testset[i][2] >= threshold: 
        real_y.append(1.0)
    else:
        real_y.append(0.0)
            
fpr, tpr, thresholds = roc_curve(real_y, est_y)
AUC = roc_auc_score(real_y, est_y)
fprsDiffModels.append(fpr)
tprsDiffModels.append(tpr)
AUCsDiffModels.append(AUC)

#%% Calculate fprs and tprs for MF with bias
from surprise.prediction_algorithms.matrix_factorization import SVD

#Best SVD on all data previously found to be k=24 using cross validation
svd = SVD(n_factors=24,verbose=False)
svd.fit(trainset)
svdPred = svd.test(testset)

    
real_y = []
est_y = []
for i in range(len(svdPred)):
    est_y.append(svdPred[i].est)
    if testset[i][2] >= threshold: 
        real_y.append(1.0)
    else:
        real_y.append(0.0)
            
fpr, tpr, thresholds = roc_curve(real_y, est_y)
AUC = roc_auc_score(real_y, est_y)
fprsDiffModels.append(fpr)
tprsDiffModels.append(tpr)
AUCsDiffModels.append(AUC)
#%% Ploting the ROC curves for all approaches
plt.figure(figsize=(12,9))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=23)
model_names = ['knn', 'nmf','mf']
for i in range(len(fprsDiffModels)):
    plt.plot(fprsDiffModels[i],tprsDiffModels[i], label = 'ROC Curve: \
             Model: {name} AUC: {auc}'.format(name=model_names[i], auc=AUCsDiffModels[i]))
plt.legend(loc="lower right")
plt.show()