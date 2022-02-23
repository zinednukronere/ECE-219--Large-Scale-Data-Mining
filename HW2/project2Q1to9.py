#%% Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
#%% Defining categories and loadig the dataset
categories = ['comp.sys.ibm.pc.hardware', 'comp.graphics',
              'comp.sys.mac.hardware', 'comp.os.ms-windows.misc',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']

dataframe = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42,remove=('headers', 'footers'))
#%% Defining the process to remove stop words and digits
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

def nothingAndRemoveDigits(textIn):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    tokenizer = CountVectorizer().build_analyzer()
    stop_words = text.ENGLISH_STOP_WORDS
    return [word for word in (tokenizer(textIn)) 
            if not any(ch.isdigit() for ch in word) and word not in stop_words]

#%% Using count vectorizer to generate the dataset
vectorize = CountVectorizer(min_df=3, 
                             analyzer=nothingAndRemoveDigits,
                             stop_words='english')
dataCountVector = vectorize.fit_transform(dataframe["data"])
#%% Getting the TF-IDF representation
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTransformer = TfidfTransformer()
data_tfidf = tfidfTransformer.fit_transform(dataCountVector)
#%%
print(data_tfidf.shape)
#%% Performing k means clustering
from sklearn.cluster import KMeans
from sklearn import metrics

km = KMeans(n_clusters=2, random_state=0, max_iter=1500, n_init=35)
km.fit(data_tfidf)
#%% Creating the contingency matrix
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score
trueClusters = [int(i/4) for i in dataframe.target]
predClusters = km.predict(data_tfidf)
con_mat = contingency_matrix(trueClusters,predClusters)
#%% Plotting contingency metrics
from plotmat import plot_mat
classes = ['Recreation', 'Computer']
plot_mat(con_mat,xticklabels = classes,yticklabels=classes,if_show_values=True,
         xlabel='Predicted label',ylabel='True label')
#%% Examining the scores
print("Homogeneity score: ", homogeneity_score(trueClusters,predClusters))
print("Completeness score: ",completeness_score(trueClusters,predClusters))
print("V-measure score: ",v_measure_score(trueClusters,predClusters))
print("Adjusted Rand score: ",adjusted_rand_score(trueClusters,predClusters))
print("Adjusted mutual information score: ",adjusted_mutual_info_score(trueClusters,predClusters))
#%% Plot the explained ratio as number of components change
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=1000,random_state = 42)
lsiTrain = lsi.fit_transform(data_tfidf)
componentNums = range(1,1001)
explainedRatio = np.sort(lsi.explained_variance_ratio_)[::-1]
explainedVars = []
for cNum in componentNums:
    explainedVars.append(sum(explainedRatio[0:cNum]))

plt.plot(componentNums,explainedVars)  
plt.xlabel("Number of components")
plt.ylabel("Explained variance ratio")  
#%% Looking at different r choices for svd
rChoices=[1, 2, 3, 5, 10, 20, 50, 100, 300]
km = KMeans(n_clusters=2, random_state=0, max_iter=1500, n_init=35)
trueClusters = [int(i/4) for i in dataframe.target]

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
#%% Plottng obtained SVD results
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
    ax.set(xlabel='Number of  Components', ylabel='Score')
fig.suptitle('Measures for SVD')
#%% Looking at different r choices for NMF
from sklearn.decomposition import NMF
rChoices=[1, 2, 3, 5, 10, 20, 50, 100, 300]
km = KMeans(n_clusters=2, random_state=0, max_iter=1500, n_init=35)
trueClusters = [int(i/4) for i in dataframe.target]

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
    
#%% Saving results so far for easier access
dictionaryNMF = {"nmfHom":nmfHom,"nmfComplt":nmfComplt,"nmfVMeas":nmfVMeas,
                 "nmfAdjRand":nmfAdjRand,"nmfAdjMutInf":nmfAdjMutInf}
import pickle
with open('dictionaryNMF.pickle', 'wb') as handle:
    pickle.dump(dictionaryNMF,handle)

#%% Loading the saved data to save time
import pickle
file = open("dictionaryNMF.pickle","rb")
dictionaryNMF = pickle.load(file)
nmfHom = dictionaryNMF["nmfHom"]
nmfComplt = dictionaryNMF["nmfComplt"]
nmfVMeas = dictionaryNMF["nmfVMeas"]
nmfAdjRand = dictionaryNMF["nmfAdjRand"]
nmfAdjMutInf = dictionaryNMF["nmfAdjMutInf"]

#%% Plotting NMF results
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
    ax.set(xlabel='Number of Components', ylabel='Score')
fig.suptitle('Measures for NMF')
#%% Plotting ground truth in 2D
import seaborn as sns

bestRSvd = 100

trueClusters = [int(i/4) for i in dataframe.target]
classes = ['Recreation', 'Computer']
svdReduced = TruncatedSVD(n_components=2,random_state=42).fit_transform(data_tfidf)
plt.figure()
for label in set(trueClusters):
    indices = [i for i in range(0,len(trueClusters)) if trueClusters[i]==label]
    plt.scatter(
    svdReduced[indices, 0],
    svdReduced[indices, 1],
    c=sns.color_palette()[label],
    label=classes[label])

plt.title("SVD Ground truth")
plt.legend()

#%% Plotting k means with SVD results in 2D
bestSvdReduced = TruncatedSVD(n_components=bestRSvd,random_state=42).fit_transform(data_tfidf)
km = KMeans(n_clusters=2, random_state=0, max_iter=1500, n_init=35)
preds = km.fit_predict(svdReduced)
plt.figure()
preds[preds==0]=2
preds[preds==1]=0
preds[preds==2]=1
for label in set(preds):
    indices = [i for i in range(0,len(preds)) if preds[i]==label]
    plt.scatter(
    svdReduced[indices, 0],
    svdReduced[indices, 1],
    c=sns.color_palette()[label])

plt.title("Kmeans results for SVD")

#%% Plotting gound truth in 2D
import seaborn as sns

bestRNmf = 2
trueClusters = [int(i/4) for i in dataframe.target]
classes = ['Recreation', 'Computer']

nmfReduced = NMF(n_components=bestRNmf,random_state=42).fit_transform(data_tfidf)
plt.figure()
for label in set(trueClusters):
    indices = [i for i in range(0,len(trueClusters)) if trueClusters[i]==label]
    plt.scatter(
    nmfReduced[indices, 0],
    nmfReduced[indices, 1],
    c=sns.color_palette()[label],
    label=classes[label])

plt.title("NMF Ground truth")
plt.legend()

#%% Plotting Kmeans with NMF results in 2D
km = KMeans(n_clusters=2, random_state=0, max_iter=1500, n_init=35)
preds = km.fit_predict(nmfReduced)
plt.figure()
preds[preds==0]=2
preds[preds==1]=0
preds[preds==2]=1
for label in set(preds):
    indices = [i for i in range(0,len(preds)) if preds[i]==label]
    plt.scatter(
    nmfReduced[indices, 0],
    nmfReduced[indices, 1],
    c=sns.color_palette()[label])

plt.title("Kmeans results for NMF")


















