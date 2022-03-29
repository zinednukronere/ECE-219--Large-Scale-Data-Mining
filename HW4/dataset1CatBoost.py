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
featurenames = ['cut','color','clarity'] + numFeats
transformedDataFrame=pd.DataFrame(transformedTrain,columns=featurenames)

#%%
from catboost import CatBoostRegressor
modelPipe = Pipeline([
    ('model',CatBoostRegressor(bagging_temperature=1.0,
             depth=9,
             iterations=775,
             l2_leaf_reg=2,
             learning_rate=0.050242625613101825,
             random_strength=1e-09))
    ])

from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedDataFrame, transformedLabels, cv=10, scoring='neg_mean_squared_error',verbose=1)
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

reg = CatBoostRegressor(verbose = False)
from skopt.space import Real, Categorical, Integer
search_spaces = {
    'iterations': Integer(10, 1000),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'depth': Integer(1, 12),
    'l2_leaf_reg': Integer(2, 100), # L2 regularization
    'random_strength': Real(1e-9, 10, 'log-uniform'), # randomness for scoring splits
    'bagging_temperature': Real(0.0, 1.0), # settings of the Bayesian bootstrap
   }

#%% Define bayes opt
from skopt import BayesSearchCV
opt = BayesSearchCV(estimator=reg,                                    
                    search_spaces=search_spaces,                      
                    scoring=scoring,                           
                    cv=10,                                           
                    n_iter=600,                                        # max number of trials
                    n_points=3,                                       # number of hyperparameter sets evaluated at the same time
                    n_jobs=-1,                                        # number of jobs
                    iid=False,                                        # if not iid it optimizes on the cv score
                    return_train_score=True,                         
                    refit=False,                                      
                    optimizer_kwargs={'base_estimator': 'GP'},        # optmizer parameters: we use Gaussian Process (GP)
                    random_state=0,
                    verbose = 10)                                   # random state for replicability

#%% Perform bayesian optimization
from skopt.callbacks import DeadlineStopper, DeltaYStopper
overdone_control = DeltaYStopper(delta=0.0001)               # We stop if the gain of the optimization becomes too small
time_limit_control = DeadlineStopper(total_time=60*60*6) # We impose a time limit (6 hours)

best_params,optimizer,d = report_perf(opt, transformedDataFrame, transformedLabels,'catboost_regression', 
                          callbacks=[overdone_control, time_limit_control])

#d.to_csv("bayesOptResForCatBoost.csv")
import pickle

#with open('bayesOptObjCatBoost.pickle', 'wb') as handle:
#    pickle.dump(optimizer,handle)


#best CV score: -0.132 ± 0.004
#Best parameters:
#OrderedDict([('bagging_temperature', 1.0), ('depth', 9), ('iterations', 775), ('l2_leaf_reg', 2), ('learning_rate', 0.050242625613101825), ('random_strength', 1e-09)])

#%%
import pickle
file = open("bayesOptObjCatBoost.pickle","rb")
loadedOptimizer = pickle.load(file)
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
#%%
#Determining effet of features
#%% Examine effect of features on performance
acceptableCols = [
       'param_bagging_temperature', 'param_depth', 'param_iterations',
       'param_l2_leaf_reg', 'param_learning_rate', 'param_random_strength',]
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['meanValScore'] = d.mean_test_score

#%%
import seaborn as sns
correlation_matrix = transformedDataFrame.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')
#%% Examine effect of features on regularization
acceptableCols = [
       'param_bagging_temperature', 'param_depth', 'param_iterations',
       'param_l2_leaf_reg', 'param_learning_rate', 'param_random_strength',]
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['absDiff'] = abs(d.mean_test_score-d.mean_train_score)
#%%
import seaborn as sns
correlation_matrix = k.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')
#%% Examine effect of features on fit efficiency

acceptableCols = [
       'param_bagging_temperature', 'param_depth', 'param_iterations',
       'param_l2_leaf_reg', 'param_learning_rate', 'param_random_strength',]
k = d.loc[:,[feat for feat in list(d.columns) if feat in acceptableCols]]
for feat in acceptableCols:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')

k['meanTrainScore'] = d.mean_train_score
#%%
import seaborn as sns
correlation_matrix = k.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, annot=True, cmap='Blues')
#%%
plt.figure()
sns.scatterplot(x=k['param_iterations'],y=k['meanValScore'])





















