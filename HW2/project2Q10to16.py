#%% Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
#%% Getting the entire dataset
dataframe = fetch_20newsgroups(subset='all',shuffle=True, random_state=42,remove=('headers', 'footers'))
#%% Defining digit and stop word removal
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

def nothingAndRemoveDigits(textIn):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    tokenizer = CountVectorizer().build_analyzer()
    stop_words = text.ENGLISH_STOP_WORDS
    return [word for word in (tokenizer(textIn)) 
            if not any(ch.isdigit() for ch in word) and word not in stop_words]
#%% Using count vectorizer to generate dataset
vectorize = CountVectorizer(min_df=3, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')
dataCountVector = vectorize.fit_transform(dataframe["data"])
#%% Generating TF-IDF representation
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTransformer = TfidfTransformer()
data_tfidf = tfidfTransformer.fit_transform(dataCountVector)
#%% Importing libraries
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
trueClusters = dataframe.target
#%% Performing truncated SVD
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=1000,random_state = 42)
lsiTrain = lsi.fit_transform(data_tfidf)
#%% Getting results for different r values for svd
rChoices=[5,20,200]
km = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target

svdHom = []
svdComplt= []
svdVMeas = []
svdAdjRand= []
svdAdjMutInf = []

for r in rChoices:
    dataToFit = lsiTrain[:,0:r]
    preds = km.fit_predict(dataToFit)
    svdHom.append(homogeneity_score(trueClusters,preds))
    svdComplt.append(completeness_score(trueClusters,preds))
    svdVMeas.append(v_measure_score(trueClusters,preds))
    svdAdjRand.append(adjusted_rand_score(trueClusters,preds))
    svdAdjMutInf.append(adjusted_mutual_info_score(trueClusters,preds))
    
#%% Plotting the results
fig,axs = plt.subplots(5)
axs[0].plot(rChoices,svdHom, 'r', label='SVD Homogeneity score')
axs[1].plot(rChoices, svdComplt, 'b', label='SVD Completeness score')
axs[2].plot(rChoices, svdVMeas, 'g', label='SVD V-measure score')
axs[3].plot(rChoices,svdAdjRand,'y',label='SVD Adjusted Rand score')
axs[4].plot(rChoices,svdAdjMutInf,'m',label='SVD Adjusted Mutual Information score')
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend()
for ax in axs.flat:
    ax.set(xlabel='No Components', ylabel='Score')
fig.suptitle('Measures for SVD')
#%% Getting results when NMF is used
from sklearn.decomposition import NMF
rChoices=[5,20,200]
km = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target

nmfHom = []
nmfComplt= []
nmfVMeas = []
nmfAdjRand= []
nmfAdjMutInf = []

for r in rChoices:
    nmf = NMF(n_components=r, init='random', random_state=42)
    nmfreduced = nmf.fit_transform(data_tfidf)
    preds = km.fit_predict(nmfreduced)
    nmfHom.append(homogeneity_score(trueClusters,preds))
    nmfComplt.append(completeness_score(trueClusters,preds))
    nmfVMeas.append(v_measure_score(trueClusters,preds))
    nmfAdjRand.append(adjusted_rand_score(trueClusters,preds))
    nmfAdjMutInf.append(adjusted_mutual_info_score(trueClusters,preds)) 
#Saving results so far for easier access
dictionaryNMF20Comp = {"nmfHom":nmfHom,"nmfComplt":nmfComplt,"nmfVMeas":nmfVMeas,
                 "nmfAdjRand":nmfAdjRand,"nmfAdjMutInf":nmfAdjMutInf}
import pickle
with open('dictionaryNMF20Comp.pickle', 'wb') as handle:
    pickle.dump(dictionaryNMF20Comp,handle)

#%% Loading the saved data to save time
import pickle
file = open("dictionaryNMF20Comp.pickle","rb")
dictionaryNMF20Comp = pickle.load(file)
nmfHom = dictionaryNMF20Comp["nmfHom"]
nmfComplt= dictionaryNMF20Comp["nmfComplt"]
nmfVMeas = dictionaryNMF20Comp["nmfVMeas"]
nmfAdjRand= dictionaryNMF20Comp["nmfAdjRand"]
nmfAdjMutInf = dictionaryNMF20Comp["nmfAdjMutInf"]
#%% Plotting the NMF results
fig,axs = plt.subplots(5)
axs[0].plot(rChoices,nmfHom, 'r', label='NMF Homogeneity score')
axs[1].plot(rChoices, nmfComplt, 'b', label='NMF Completeness score')
axs[2].plot(rChoices, nmfVMeas, 'g', label='NMF V-measure score')
axs[3].plot(rChoices,nmfAdjRand,'y',label='NMF Adjusted Rand score')
axs[4].plot(rChoices,nmfAdjMutInf,'m',label='NMF Adjusted Mutual Information score')
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend()
for ax in axs.flat:
    ax.set(xlabel='No Components', ylabel='Score')
fig.suptitle('Measures for NMF')
#%% Compaing results obtained with SVD and NMF
bestVMeasSVD = np.amax(np.array(svdVMeas)) #20 components
bestVMeasNMF = np.amax(np.array(nmfVMeas))
#SVD is better with bestSVD=200
#%% Defining best number of components for svd
#bestRSVD = rChoices[np.argmax(np.array(svdVMeas))] #200
bestRSVD=200
#%% Dim reduction with svd
svdReduced = TruncatedSVD(n_components=bestRSVD,random_state=42).fit_transform(data_tfidf)
#%% Performing k means clustering
km = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
preds = km.fit_predict(svdReduced)
trueClusters = dataframe.target
#%% Plotting contingency matrix after linear optimisation
import numpy as np
from plotmat import plot_mat 
from scipy.optimize import linear_sum_assignment
cm = contingency_matrix(trueClusters, preds)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15),
         xlabel='Predicted label',ylabel='True label')
#%% Printing the clustering measures
print("Homogeneity score: ", homogeneity_score(trueClusters,preds))
print("Completeness score: ",completeness_score(trueClusters,preds))
print("V-measure score: ",v_measure_score(trueClusters,preds))
print("Adjusted Rand score: ",adjusted_rand_score(trueClusters,preds))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(trueClusters,preds))
#%%
#################################################
#UMAP
#################################################

#%% Getting UMAP results with different components
import umap.umap_ as umap
rChoices=[5,20,200]
km = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target

UMAPHom = []
UMAPComplt= []
UMAPVMeas = []
UMAPAdjRand= []
UMAPAdjMutInf = []

for r in rChoices:
    reducer = umap.UMAP(n_components=r)
    UMAPreduced = reducer.fit_transform(data_tfidf)
    preds = km.fit_predict(UMAPreduced)
    UMAPHom.append(homogeneity_score(trueClusters,preds))
    UMAPComplt.append(completeness_score(trueClusters,preds))
    UMAPVMeas.append(v_measure_score(trueClusters,preds))
    UMAPAdjRand.append(adjusted_rand_score(trueClusters,preds))
    UMAPAdjMutInf.append(adjusted_mutual_info_score(trueClusters,preds)) 
#Saving results so far for easier access
dictionaryUMAP20Comp = {"UMAPHom":UMAPHom,"UMAPComplt":UMAPComplt,"UMAPVMeas":UMAPVMeas,
                 "UMAPAdjRand":UMAPAdjRand,"UMAPAdjMutInf":UMAPAdjMutInf}
import pickle
with open('dictionaryUMAP20Comp.pickle', 'wb') as handle:
    pickle.dump(dictionaryUMAP20Comp,handle)
    
#%% Loading the saved data to save time
import pickle
file = open("dictionaryUMAP20Comp.pickle","rb")
dictionaryUMAP20Comp = pickle.load(file)
UMAPHom = dictionaryUMAP20Comp["UMAPHom"]
UMAPComplt= dictionaryUMAP20Comp["UMAPComplt"]
UMAPVMeas = dictionaryUMAP20Comp["UMAPVMeas"]
UMAPAdjRand= dictionaryUMAP20Comp["UMAPAdjRand"]
UMAPAdjMutInf = dictionaryUMAP20Comp["UMAPAdjMutInf"]
#%% Getting the best UMAP dimension
bestRUMAP = rChoices[np.argmax(np.array(UMAPVMeas))] #It is 200
#%%
###################################
#EUCLIDEAN UMAP
###################################
#%% Using Euclidean UMAP for dim reduction and k-means for clustering
kmEuc = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target
reducerEuc = umap.UMAP(n_components=200,metric="euclidean")
UMAPreducedEuc = reducerEuc.fit_transform(data_tfidf)
predsEuc = kmEuc.fit_predict(UMAPreducedEuc)
#%% Plot contingency matrix
from plotmat import plot_mat 
from scipy.optimize import linear_sum_assignment
cm = contingency_matrix(trueClusters, predsEuc)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15),
         xlabel='Predicted label',ylabel='True label')
#%%
print("Homogeneity score Euclidean: ", homogeneity_score(trueClusters,predsEuc))
print("Completeness score Euclidean: ",completeness_score(trueClusters,predsEuc))
print("V-measure score Euclidean: ",v_measure_score(trueClusters,predsEuc))
print("Adjusted Rand score Euclidean: ",adjusted_rand_score(trueClusters,predsEuc))
print("Adjusted mutual information score Euclidean: ",adjusted_mutual_info_score(trueClusters,predsEuc))
#%%
###################################
#COSINE UMAP
###################################
#%% Using Cosine UMAP for dim reduction and k-means for clustering
kmCos = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target
reducerCos = umap.UMAP(n_components=200,metric="cosine")
UMAPreducedCos = reducerCos.fit_transform(data_tfidf)
predsCos = kmCos.fit_predict(UMAPreducedCos)
#%% Plot contingency matrix
from plotmat import plot_mat 
from scipy.optimize import linear_sum_assignment
cm = contingency_matrix(trueClusters, predsCos)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15),
         xlabel='Predicted label',ylabel='True label')
#%%
print("Homogeneity score Euclidean: ", homogeneity_score(trueClusters,predsCos))
print("Completeness score Euclidean: ",completeness_score(trueClusters,predsCos))
print("V-measure score Euclidean: ",v_measure_score(trueClusters,predsCos))
print("Adjusted Rand score Euclidean: ",adjusted_rand_score(trueClusters,predsCos))
print("Adjusted mutual information score Euclidean: ",adjusted_mutual_info_score(trueClusters,predsCos))
#%% Fit KNN directly to the dataset
km = KMeans(n_clusters=20, random_state=0, max_iter=1500, n_init=35)
trueClusters = dataframe.target
preds = km.fit_predict(data_tfidf)
#%% Plot contingency matrix
from plotmat import plot_mat 
from scipy.optimize import linear_sum_assignment
cm = contingency_matrix(trueClusters, predsCos)
rows, cols = linear_sum_assignment(cm, maximize=True)
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=cols,yticklabels=rows, size=(15,15),
         xlabel='Predicted label',ylabel='True label')
#%%
print("Homogeneity score : ", homogeneity_score(trueClusters,preds))
print("Completeness score : ",completeness_score(trueClusters,preds))
print("V-measure score : ",v_measure_score(trueClusters,preds))
print("Adjusted Rand score : ",adjusted_rand_score(trueClusters,preds))
print("Adjusted mutual information score : ",adjusted_mutual_info_score(trueClusters,preds))

#%%
#####################################
#AGGLOMERATIVE CLUSTERING
#####################################
#%% Use UMAP to reduce dimension
reducerCos = umap.UMAP(n_components=200,metric="cosine")
UMAPreducedCos = reducerCos.fit_transform(data_tfidf)
#%% Defining agglo cluster with ward linkage
from sklearn.cluster import AgglomerativeClustering
clusteringWard = AgglomerativeClustering(n_clusters=20,linkage="ward")
predsAggloWard = clusteringWard.fit_predict(UMAPreducedCos)
#%% Checking performance
trueClusters = dataframe.target
print("Homogeneity score clusteringWard: ", homogeneity_score(trueClusters,predsAggloWard))
print("Completeness score clusteringWard: ",completeness_score(trueClusters,predsAggloWard))
print("V-measure score clusteringWard: ",v_measure_score(trueClusters,predsAggloWard))
print("Adjusted Rand score clusteringWard: ",adjusted_rand_score(trueClusters,predsAggloWard))
print("Adjusted mutual information score clusteringWard: ",adjusted_mutual_info_score(trueClusters,predsAggloWard))
#%% Defining single linkage agglo
clusteringSingle = AgglomerativeClustering(n_clusters=20,linkage="single")
predsAggloSingle = clusteringSingle.fit_predict(UMAPreducedCos)
#%% Check performance
trueClusters = dataframe.target
print("Homogeneity score clusteringSingle: ", homogeneity_score(trueClusters,predsAggloSingle))
print("Completeness score clusteringSingle: ",completeness_score(trueClusters,predsAggloSingle))
print("V-measure score clusteringSingle: ",v_measure_score(trueClusters,predsAggloSingle))
print("Adjusted Rand score clusteringSingle: ",adjusted_rand_score(trueClusters,predsAggloSingle))
print("Adjusted mutual information score clusteringSingle: ",adjusted_mutual_info_score(trueClusters,predsAggloSingle))
#%%
####################################
#DBSCAN
###################################

#%% Use UMAP to reduce dimension
reducerCos = umap.UMAP(n_components=200,metric="cosine")
UMAPreducedCos = reducerCos.fit_transform(data_tfidf)
#%% Define dbscan
from sklearn.cluster import DBSCAN
dbscanner = DBSCAN(eps=0.5)
predsdbscan = dbscanner.fit_predict(UMAPreducedCos)

#%% Check performance
trueClusters = dataframe.target
print("Homogeneity score dbscanner: ", homogeneity_score(trueClusters,predsdbscan))
print("Completeness score dbscanner: ",completeness_score(trueClusters,predsdbscan))
print("V-measure score dbscanner: ",v_measure_score(trueClusters,predsdbscan))
print("Adjusted Rand score dbscanner: ",adjusted_rand_score(trueClusters,predsdbscan))
print("Adjusted mutual information score dbscanner: ",adjusted_mutual_info_score(trueClusters,predsdbscan))
#%% Define hdbscan
import hdbscan
hdbscanner = hdbscan.HDBSCAN(min_cluster_size=100)
predshdbscan = hdbscanner.fit_predict(UMAPreducedCos)
#%%Check performance
trueClusters = dataframe.target
print("Homogeneity score dbscanner: ", homogeneity_score(trueClusters,predshdbscan))
print("Completeness score dbscanner: ",completeness_score(trueClusters,predshdbscan))
print("V-measure score dbscanner: ",v_measure_score(trueClusters,predshdbscan))
print("Adjusted Rand score dbscanner: ",adjusted_rand_score(trueClusters,predshdbscan))
print("Adjusted mutual information score dbscanner: ",adjusted_mutual_info_score(trueClusters,predshdbscan))
#%% set the best model prediction after comparing dbscan and hdbscan
bestPreds=predshdbscan
#%% Plot contingency matrix
trueClusters = dataframe.target
from plotmat import plot_mat 
from scipy.optimize import linear_sum_assignment
cm = contingency_matrix(trueClusters, bestPreds)
setPreds = list(set(bestPreds))
rows, cols = linear_sum_assignment(cm, maximize=True)
xLabels = [setPreds[i] for i in cols]
plot_mat(cm[rows[:, np.newaxis], cols], xticklabels=xLabels,yticklabels=rows, size=(15,15),
         xlabel='Predicted label',ylabel='True label')


