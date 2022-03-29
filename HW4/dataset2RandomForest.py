#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
data = pd.read_csv("combinedEmissionData.csv")
data.drop('Unnamed: 0',axis = 1,inplace=True)
data.drop('NOX',axis = 1,inplace=True)
#%% Get info about the data
print(data.info())
#No null values
#%% Seperate to train and test
import random
from sklearn.model_selection import train_test_split
np.random.seed(42)
random.seed(42)
train, test = train_test_split(data, test_size=0.2)
#%% Seperate to features and labels
featuresTrain = train.loc[:,[feat for feat in list(train.columns) if feat!="CO"]]
labelsTrain = train.CO
#%% Pipeline for feature processing
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ordenYear = OrdinalEncoder(categories=[[2011,2012,2013,2014,2015]])

numFeats = list(featuresTrain.select_dtypes(include=['float64']).columns)
numScale =  StandardScaler()
preprocessColumns = ColumnTransformer(
    transformers=[
        ("ordYear", ordenYear, ["Year"]),
    ],
    remainder="passthrough"
)
featureProcessPipe = Pipeline([('A',preprocessColumns),('B',numScale)])
noScalefeatureProcessPipe = Pipeline([('A',preprocessColumns)])
labelProcessor = StandardScaler()
#%% Pipeline for feature processing
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

featurePipe = Pipeline([
    ('columnTransform',featureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ])
#%% Transforming data and labels 
transformedLabels = labelProcessor.fit_transform(np.array(labelsTrain).reshape(-1,1)).ravel()
transformedTrain = featurePipe.fit_transform(featuresTrain,transformedLabels)
featurenames = ['Year'] + numFeats
transformedTrainDataFrame=pd.DataFrame(transformedTrain,columns=featurenames)
#%%
from sklearn.ensemble import RandomForestRegressor
modelPipe = Pipeline([
    ('model',RandomForestRegressor(max_depth=100, max_features=7, n_estimators=200))
    ])
#%%
from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, 
                            scoring='neg_mean_squared_error',verbose=10,return_train_score=True)
meanCVTrainScore = np.mean(cv_results['train_score'])
meanCVTestScore = np.mean(cv_results['test_score'])
print(meanCVTrainScore) 
print(meanCVTestScore) 
#%%
from sklearn.metrics import mean_squared_error
model = modelPipe.fit(transformedTrain, transformedLabels)
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="CO"]]
labelsTest = test.CO
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = model.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Getting oob score
from sklearn.metrics import mean_squared_error
model = RandomForestRegressor(max_depth=100, max_features=7, n_estimators=200,oob_score=True)
trainedModel = model.fit(transformedTrain,transformedLabels)
oob_error = 1 - trainedModel.oob_score_

#%% Define param to search
paramGrid = [
     {
          "n_estimators":[50,100,150,200],
          "max_features":[3,5,7,9,10],
          "max_depth":[10,100,500,1000,2500,None]

     }
]
#%% Complete search
from sklearn.model_selection import GridSearchCV
est = RandomForestRegressor()
grid = GridSearchCV(est, paramGrid,cv=10,n_jobs=1, scoring='neg_mean_squared_error',
                    verbose=10,return_train_score=True)
grid.fit(transformedTrain, transformedLabels)
results = pd.DataFrame(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
import pickle

with open('gridResultsRandomForestData2.pickle', 'wb') as handle:
    pickle.dump(grid,handle)
#-0.018527770723580316
#{'model': RandomForestRegressor(max_depth=100, max_features=7, n_estimators=200), 'model__max_depth': 100, 'model__max_features': 7, 'model__n_estimators': 200}
#%%
import pickle

#with open('gridResultsRandomForestData2.pickle', 'wb') as handle:
#    pickle.dump(grid,handle)
#%%
import pickle
file = open("gridResultsRandomForestData2.pickle","rb")
grid2 = pickle.load(file)
results = pd.DataFrame(grid2.cv_results_)
#%% Gettest rms
from sklearn.metrics import mean_squared_error
bestModel = grid.best_estimator_
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="CO"]]
labelsTest = test.CO
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = bestModel.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)

#%% Determine effect of features
desiredfeats=['param_max_depth', 'param_max_features',
       'param_n_estimators','mean_test_score']
k = results.loc[:,[feat for feat in desiredfeats]]
for feat in desiredfeats:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')
#%%
import seaborn as sns
correlation_matrix = k.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', annot=True, cmap='Blues')
#%%
#VISUALISE TREE
#%% Fit new tree
model = RandomForestRegressor(max_depth=4, max_features=7, n_estimators=200,oob_score=True)
trainedModel = model.fit(transformedTrain,transformedLabels)

#%%
depths = [estimator.tree_.max_depth for estimator in model.estimators_]
#%% Visualise
from sklearn import tree
import random
featurenames = ['Year'] + numFeats
transformedDataFrame=pd.DataFrame(transformedTrain,columns=featurenames)
treeIndex = random.randrange(len(model.estimators_)+1)
#rf = RandomForestRegressor(n_estimators=50,max_depth=4)
#rf.fit(transformedDataFrame,transformedLabels)
plt.figure(figsize=(10,10))
_ = tree.plot_tree(model.estimators_[treeIndex], feature_names=transformedDataFrame.columns, filled=True,fontsize=10)

for name, importance in zip(transformedDataFrame.columns, model.estimators_[treeIndex].feature_importances_):
    print(name, importance)




















