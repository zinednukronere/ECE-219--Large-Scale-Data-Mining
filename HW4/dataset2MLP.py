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
from sklearn.neural_network import MLPRegressor
modelPipe = Pipeline([
    ('model',MLPRegressor(alpha=0.001, hidden_layer_sizes=(100, 150)))
    ])
#%% Performing cross validation on MLP
from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, 
                            scoring='neg_mean_squared_error',verbose=10,return_train_score=True)
meanCVTrainScore = np.mean(cv_results['train_score'])
meanCVTestScore = np.mean(cv_results['test_score'])
print(meanCVTrainScore) 
print(meanCVTestScore) 

#%%
paramGrid = [
     {
          "model":[MLPRegressor()],
          "model__hidden_layer_sizes":[(50),(100),(150),(50,50),(50,100),(100,150),(50,150)],
          "model__alpha":[0.00001,0.0001,0.001,0.01,0.1]

     }
]
#%% Performing grid search
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(modelPipe, paramGrid,cv=10,n_jobs=1, scoring='neg_mean_squared_error',
                    verbose=10,return_train_score=True)
grid.fit(transformedTrain, transformedLabels)
results = pd.DataFrame(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
#-0.2147488832145507
#{'model': MLPRegressor(alpha=0.001, hidden_layer_sizes=(100, 150)), 'model__alpha': 0.001, 'model__hidden_layer_sizes': (100, 150)}
#%%
import pickle

# with open('gridResultsData2MLP.pickle', 'wb') as handle:
#     pickle.dump(grid,handle)
    
#%%
import pickle
file = open("gridResultsData2MLP.pickle","rb")
grid = pickle.load(file)
results = pd.DataFrame(grid.cv_results_)
#%% Getting test error
from sklearn.metrics import mean_squared_error
model = modelPipe.fit(transformedTrain, transformedLabels)
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="CO"]]
labelsTest = test.CO
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = model.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#0.5076729552520788


