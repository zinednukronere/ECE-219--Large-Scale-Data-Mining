#%%Generating the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import pickle
import umap.umap_ as umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import hdbscan
#%% Getting the data, generating TF-IDF set
dataframe = fetch_20newsgroups(subset='all',shuffle=True, random_state=42,remove=('headers', 'footers'))

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

def nothingAndRemoveDigits(textIn):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    tokenizer = CountVectorizer().build_analyzer()
    stop_words = text.ENGLISH_STOP_WORDS
    return [word for word in (tokenizer(textIn)) 
            if not any(ch.isdigit() for ch in word) and word not in stop_words]

vectorize = CountVectorizer(min_df=3, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')
dataCountVector = vectorize.fit_transform(dataframe["data"])

from sklearn.feature_extraction.text import TfidfTransformer
tfidfTransformer = TfidfTransformer()
data_tfidf = tfidfTransformer.fit_transform(dataCountVector)
#%%Performing dimensionality reduction early because doing it in a loop will take literal years
svd5 = TruncatedSVD(n_components=5,random_state = 42)
svd5Data = svd5.fit_transform(data_tfidf)
svd20 = TruncatedSVD(n_components=20,random_state = 42)
svd20Data = svd20.fit_transform(data_tfidf)
svd200 = TruncatedSVD(n_components=200,random_state = 42)
svd200Data = svd200.fit_transform(data_tfidf)
#%%
nmf5 = NMF(n_components=5, init='random', random_state=42)
nmf5Data = nmf5.fit_transform(data_tfidf)
nmf20 = NMF(n_components=20, init='random', random_state=42)
nmf20Data = nmf20.fit_transform(data_tfidf)
nmf200 = NMF(n_components=200, init='random', random_state=42)
nmf200Data = nmf200.fit_transform(data_tfidf)
#%%
umap5 = umap.UMAP(n_components=5,metric="cosine")
umap5Data = umap5.fit_transform(data_tfidf)
umap20 = umap.UMAP(n_components=20,metric="cosine")
umap20Data = umap20.fit_transform(data_tfidf)
umap200 = umap.UMAP(n_components=200,metric="cosine")
umap200Data = umap200.fit_transform(data_tfidf)
#%% Save the results
dimReduceDict={"svd5Data":svd5Data,"svd20Data":svd20Data,"svd200Data":svd200Data,
               "nmf5Data":nmf5Data,"nmf20Data":nmf20Data,"nmf200Data":nmf200Data,
               "umap5Data":umap5Data,"umap20Data":umap20Data,"umap200Data":umap200Data}
with open('dictionaryDimReduce.pickle', 'wb') as handle:
    pickle.dump(dimReduceDict,handle)
#%% Loading the results
import pickle
file = open("dictionaryDimReduce.pickle","rb")
dimReduceDict = pickle.load(file)
#%%
##############################################
#PARAMETER SEARCH
##############################################
#%% Setting up the search
dimReduceDict.update({"None": data_tfidf})
trueClusters = dataframe.target

clustering = ["kmeans","agglo","dbscan","hdbscan"]
kMeansParams=[10,20,50]
eps = [0.5,5] 
minClusterSizes = [100,200]


i=0

usedData = []
usedModel = []
homoScores = []
completeScores = []
vMeasScores = []
adjRandScores = []
mutInfoScores  = []
#%%Performing grid parameter search
keys =dimReduceDict.keys()
clustering = ["kmeans","agglo","dbscan","hdbscan"]
for key in keys:
    dataToFit = dimReduceDict[key]
    for model in clustering:
        if model=="kmeans":
            for k in kMeansParams:
                km = KMeans(n_clusters=k, random_state=0, max_iter=1500, n_init=35)
                preds = km.fit_predict(dataToFit)
                currModel = "kmeansWithk="+str(k)
                homo = homogeneity_score(trueClusters,preds)
                complete=completeness_score(trueClusters,preds)
                vMeas= v_measure_score(trueClusters,preds)
                adjRand=adjusted_rand_score(trueClusters,preds)
                mutInf=adjusted_mutual_info_score(trueClusters,preds)
                
                usedData.append(key)
                usedModel.append(currModel)
                homoScores.append(homo)
                completeScores.append(complete)
                vMeasScores.append(vMeas)
                adjRandScores.append(adjRand)
                mutInfoScores.append(mutInf)  
                
                print("Completed iteration "+str(i+1)+" out of 80")
                i=i+1
                
        elif model=="agglo":
            dataToFit = dataToFit.toarray()
            clusteringWard = AgglomerativeClustering(n_clusters=20,linkage="ward")
            preds = clusteringWard.fit_predict(dataToFit)
            currModel = "agglomerativeClustering"
            homo = homogeneity_score(trueClusters,preds)
            complete=completeness_score(trueClusters,preds)
            vMeas= v_measure_score(trueClusters,preds)
            adjRand=adjusted_rand_score(trueClusters,preds)
            mutInf=adjusted_mutual_info_score(trueClusters,preds)
            
            usedData.append(key)
            usedModel.append(currModel)
            homoScores.append(homo)
            completeScores.append(complete)
            vMeasScores.append(vMeas)
            adjRandScores.append(adjRand)
            mutInfoScores.append(mutInf)  
            
            print("Completed iteration "+str(i+1)+" out of 80")
            i=i+1
            
        elif model=="dbscan":
            for ep in eps:
                dbscanner = DBSCAN(eps=ep)
                preds = dbscanner.fit_predict(dataToFit)
                currModel = "dbscanWitheps="+str(ep)
                homo = homogeneity_score(trueClusters,preds)
                complete=completeness_score(trueClusters,preds)
                vMeas= v_measure_score(trueClusters,preds)
                adjRand=adjusted_rand_score(trueClusters,preds)
                mutInf=adjusted_mutual_info_score(trueClusters,preds)

                usedData.append(key)
                usedModel.append(currModel)
                homoScores.append(homo)
                completeScores.append(complete)
                vMeasScores.append(vMeas)
                adjRandScores.append(adjRand)
                mutInfoScores.append(mutInf)  
                
                print("Completed iteration "+str(i+1)+" out of 80")
                i=i+1
                
        elif model=="hdbscan":
            for clusterSize in minClusterSizes:
                hdbscanner = hdbscan.HDBSCAN(min_cluster_size=clusterSize)
                preds = hdbscanner.fit_predict(dataToFit)
                currModel = "hdbScannerWithClusterSize="+str(clusterSize)
                homo = homogeneity_score(trueClusters,preds)
                complete=completeness_score(trueClusters,preds)
                vMeas= v_measure_score(trueClusters,preds)
                adjRand=adjusted_rand_score(trueClusters,preds)
                mutInf=adjusted_mutual_info_score(trueClusters,preds)

                usedData.append(key)
                usedModel.append(currModel)
                homoScores.append(homo)
                completeScores.append(complete)
                vMeasScores.append(vMeas)
                adjRandScores.append(adjRand)
                mutInfoScores.append(mutInf)          
        
                print("Completed iteration "+str(i+1)+" out of 80")
                i=i+1
#%%
#COULDNT COMPLETE NO DIM REDUCTION HDBSCAN AND AGGLO
#%% Saving results so far
resultsDict={"usedData":usedData,"usedModel":usedModel,"homoScores":homoScores,
               "completeScores":completeScores,"vMeasScores":vMeasScores,"adjRandScores":adjRandScores,
               "mutInfoScores":mutInfoScores}
with open('resultsDictionary.pickle', 'wb') as handle:
    pickle.dump(resultsDict,handle)

#%% Loading the obtained results
import pickle
file = open("resultsDictionary.pickle","rb")
resultsDictionary = pickle.load(file)
usedData = resultsDictionary["usedData"]
usedModel = resultsDictionary["usedModel"]
homoScores = resultsDictionary["homoScores"]
completeScores = resultsDictionary["completeScores"]
vMeasScores = resultsDictionary["vMeasScores"]
adjRandScores = resultsDictionary["adjRandScores"]
mutInfoScores = resultsDictionary["mutInfoScores"]
#%% Getting the best 5 scores and combinations
sortIndices = np.argsort(adjRandScores)[::-1]
bestV = sortIndices[0:5]
bestData = [usedData[k] for k in bestV]
bestModel = [usedModel[k] for k in bestV]
bestHomo = [homoScores[k] for k in bestV]
bestComplete = [completeScores[k] for k in bestV]
bestVMeas = [vMeasScores[k] for k in bestV]
bestAdjRand = [adjRandScores[k] for k in bestV]
bestMutInfo = [mutInfoScores[k] for k in bestV]









