#%% Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Load the data
data = pd.read_csv("diamonds.csv")
data.drop('Unnamed: 0',axis = 1,inplace=True)
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
featuresTrain = train.loc[:,[feat for feat in list(train.columns) if feat!="price"]]
labelsTrain = train.price
#%% Pipeline for feature processing
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ordenCut = OrdinalEncoder(categories=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']])
ordenColor = OrdinalEncoder(categories=[['J', 'I', 'H', 'G', 'F', 'E', 'D']])
ordenClarity = OrdinalEncoder(categories=[['I1', 'SI2', 'SI1', 'VS2','VS1', 'VVS2', 
                                           'VVS1', 'IF']])

numFeats = list(data.select_dtypes(include=['float64']).columns)
numScale =  StandardScaler()
preprocessColumns = ColumnTransformer(
    transformers=[
        ("ordCut", ordenCut, ["cut"]),
        ("ordColor", ordenColor, ["color"]),
        ("ordClarity", ordenClarity, ["clarity"])
    ],
    remainder="passthrough"
)
featureProcessPipe = Pipeline([('A',preprocessColumns),('B',numScale)])
labelProcessor = StandardScaler()
#%% Pipeline for feature processing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

featurePipe = Pipeline([
    ('columnTransform',featureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ])
#%% Transforming data and labels 
transformedLabels = labelProcessor.fit_transform(np.array(labelsTrain).reshape(-1,1)).ravel()
transformedTrain = featurePipe.fit_transform(featuresTrain,transformedLabels)
#%% 
from lightgbm import LGBMRegressor
modelPipe = Pipeline([
    ('model',LGBMRegressor(colsample_bytree= 0.9145036000614722,
             learning_rate= 0.025462695402919337,
             max_depth= 63,
             n_estimators= 800,
             num_leaves= 123,
             reg_alpha= 1.866498939433158,
             reg_lambda=5.975154986953605e-06,
             reg_sqrt= False,
             subsample= 1.0,
             subsample_freq=9))
    ])

from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, scoring='neg_mean_squared_error',verbose=1)
meanCVScore = np.mean(cv_results['test_score'])
print(meanCVScore)
#%% Getting test performance of default mdoel
from sklearn.metrics import mean_squared_error
learnedModels = modelPipe.fit(transformedTrain, transformedLabels)
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = learnedModels.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%%
from time import time

def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1"+" %.3f") % (time() - start, 
                                   len(optimizer.cv_results_['params']),
                                   best_score,
                                   best_score_std))    
    print('Best parameters:')
    print(best_params)
    print()
    return best_params,optimizer,d
#%%
from sklearn.metrics import make_scorer
from functools import partial
scoring = make_scorer(partial(mean_squared_error, squared=False), 
                      greater_is_better=False)
#%% Defining search space
reg = LGBMRegressor(random_state=0)
from skopt.space import Real, Categorical, Integer

search_spaces = {
    'reg_sqrt': Categorical([True, False]),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),     # Boosting learning rate
    'n_estimators': Integer(30, 800),                   # Number of boosted trees to fit
    'num_leaves': Integer(2, 512),                       # Maximum tree leaves for base learners
    'max_depth': Integer(-1, 256),                       # Maximum tree depth for base learners, <=0 means no limit
    'subsample': Real(0.01, 1.0, 'uniform'),             # Subsample ratio of the training instance
    'subsample_freq': Integer(1, 10),                    # Frequency of subsample, <=0 means no enable
    'colsample_bytree': Real(0.01, 1.0, 'uniform'),      # Subsample ratio of columns when constructing each tree
    'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),      # L2 regularization
    'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),       # L1 regularization
   }
#%% Define bayes opt
from skopt import BayesSearchCV
opt = BayesSearchCV(estimator=reg,                                    
                    search_spaces=search_spaces,                      
                    scoring=scoring,                           
                    cv=10,                                           
                    n_iter=300,                                        # max number of trials
                    n_points=3,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs=-1,                                        # number of jobs
                    iid=False,                                        # if not iid it optimizes on the cv score
                    return_train_score=True,                         
                    refit=False,                                      
                    optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state=0,
                    verbose = 2)                                   # random state for replicability

#%% Perform bayesian optimization
from skopt.callbacks import DeadlineStopper, DeltaYStopper
import pickle

overdone_control = DeltaYStopper(delta=0.0001)               # We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*6) # We impose a time limit (6 hours)

best_params,optimizer,d = report_perf(opt, transformedTrain, transformedLabels,'LightGBM_regression', 
                          callbacks=[overdone_control, time_limit_control])

#d.to_csv("bayesOptResForLightGBM.csv")

#with open('bayesOptObjLgbm.pickle', 'wb') as handle:
#   pickle.dump(optimizer,handle)

#%%
import pickle
file = open("bayesOptObjLgbm.pickle","rb")
loadedOptimizer = pickle.load(file)

#%%
d = pd.DataFrame(loadedOptimizer.cv_results_)

#%% Get RMSE on test set
from sklearn.metrics import mean_squared_error
bestModel = loadedOptimizer.best_estimator_
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = bestModel.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Examine effect of features on performance
acceptableCols = [
       'param_colsample_bytree', 'param_learning_rate', 'param_max_depth',
       'param_n_estimators', 'param_num_leaves', 'param_reg_alpha',
       'param_reg_lambda', 'param_reg_sqrt', 'param_subsample',
       'param_subsample_freq']
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['meanValScore'] = d.mean_test_score

import seaborn as sns
correlation_matrix = k.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')
#%% Examine effect of features on regularization
acceptableCols = [
       'param_colsample_bytree', 'param_learning_rate', 'param_max_depth',
       'param_n_estimators', 'param_num_leaves', 'param_reg_alpha',
       'param_reg_lambda', 'param_reg_sqrt', 'param_subsample',
       'param_subsample_freq']
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['absDiff'] = abs(d.mean_test_score-d.mean_train_score)

import seaborn as sns
correlation_matrix = k.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')

#%% Examine effect of features on fit efficiency
acceptableCols = [
       'param_colsample_bytree', 'param_learning_rate', 'param_max_depth',
       'param_n_estimators', 'param_num_leaves', 'param_reg_alpha',
       'param_reg_lambda', 'param_reg_sqrt', 'param_subsample',
       'param_subsample_freq']
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['meanTrainScore'] = d.mean_train_score

import seaborn as sns
correlation_matrix = k.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')


#%%
sns.scatterplot(x=k['param_n_estimators'],y=k['mean_test_score'])




