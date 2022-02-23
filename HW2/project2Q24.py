import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix, homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score


#%% Dimensionality Reduction

#Load vgg156 flower features
vgg_features = np.load('./flowers_features_and_labels.npz')
f_all = vgg_features['f_all']
y_all = vgg_features['y_all']

# SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50,random_state = 42)
svd_features = svd.fit_transform(f_all)

#UMAP
import umap.umap_ as umap
reducer = umap.UMAP(n_components=50)
umap_features = reducer.fit_transform(f_all)

# Autoencoder
auto_features = np.load('./autoencoder_features.npy')

features = [f_all, svd_features, umap_features, auto_features]

#%% Search for kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, random_state=0, max_iter=1500, n_init=35)

from sklearn.cluster import AgglomerativeClustering
aggCluster = AgglomerativeClustering(n_clusters=5,linkage="ward")

import hdbscan

km_scores = {}
dim_red_names = ['None', 'SVD', 'UMAP', 'Autoencoder']

for feature_set, name in zip(features, dim_red_names):
    km_preds = km.fit_predict(feature_set)
    agg_preds = aggCluster.fit_predict(feature_set)
    km_score = adjusted_rand_score(y_all,km_preds)
    agg_score = adjusted_rand_score(y_all,agg_preds)
    print("Adjusted Rand score " + name + ' km: ',km_score)
    print("Adjusted Rand score " + name + ' agg: ',agg_score)
    
#%% Grid Search for hdbscan
for feature_set, name in zip(features, dim_red_names):
    for mc in [5,10,15,20,30,50]:
        for ms in [1,5,10,15,20,50]:
            if ms > mc:
                pass
            else:
                hdbscanner = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=ms)
                hdbscan_preds = hdbscanner.fit_predict(feature_set)
                hdbscan_score = adjusted_rand_score(y_all,hdbscan_preds)
                print("Adjusted Rand score " + name + ' min cluster: '+ str(mc) + ' min samples: '+ str(ms) + ' HDB: ',hdbscan_score)
#%% Grid Search for hdbscan pt 2.
for feature_set, name in zip(features, dim_red_names):
    for mc in [80,120, 200]:
        for ms in [10,15,20]:
            if ms > mc:
                pass
            else:
                hdbscanner = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=ms)
                hdbscan_preds = hdbscanner.fit_predict(feature_set)
                hdbscan_score = adjusted_rand_score(y_all,hdbscan_preds)
                print("Adjusted Rand score " + name + ' min cluster: '+ str(mc) + ' min samples: '+ str(ms) + ' HDB: ',hdbscan_score)

#%% Saving dimensionality reduced features
np.save('./UMAP_features.npy',umap_features)
np.save('./SVD_features.npy',svd_features)
np.save('./Autoencoder_features.npy',auto_features)


