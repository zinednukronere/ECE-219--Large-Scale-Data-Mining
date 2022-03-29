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
#%% Model pipeline for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
modelPipe = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('model',LinearRegression())
    ])
#%% Cross validating model
from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, 
                            scoring='neg_mean_squared_error',verbose=1,return_train_score=True)
meanCVScore = np.mean(cv_results['test_score'])
print(meanCVScore)
meanCVScore = np.mean(cv_results['train_score'])
print(meanCVScore)
#%% Determining grid to search over
paramGrid = [
     {
          "poly":[PolynomialFeatures()],
          "poly__degree":[2, 3, 4, 5]

     }
]
#%% Performing search
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(modelPipe, paramGrid,cv=10,n_jobs=1, scoring='neg_mean_squared_error',
                    verbose=10,return_train_score=True)
grid.fit(transformedTrain, transformedLabels)
results = pd.DataFrame(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
#Best result obtained when polynomial degree is 2
#%% Getting test rms
from sklearn.metrics import mean_squared_error
bestModel = grid.best_estimator_
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = bestModel.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% ransform data
#Degree 2 is good. Look up the salient features
poly = PolynomialFeatures(degree=2)
degreeTransform = poly.fit_transform(transformedTrain)
#%% Determining feature importance
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

featureFilterFRegression = SelectKBest(f_regression, k='all')
newTrainFReg = featureFilterFRegression.fit_transform(degreeTransform,transformedLabels)
print(featureFilterFRegression.scores_)
#%%
#############################################3
#%% Creating different features
data = pd.read_csv("diamonds.csv")
data.drop('Unnamed: 0',axis = 1,inplace=True)
data['volume'] = data.x * data.y * data.z
#data.drop('x',axis = 1,inplace=True)
#data.drop('y',axis = 1,inplace=True)
#data.drop('z',axis = 1,inplace=True)
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
#%% Model pipeline for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
modelPipe = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('model',LinearRegression())
    ])
#%% Getting cv scores with new feature
from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, 
                            scoring='neg_mean_squared_error',verbose=1,return_train_score=True)
meanCVTrainScore = np.mean(cv_results['train_score'])
meanCVTestScore = np.mean(cv_results['test_score'])
print(meanCVTrainScore)
print(meanCVTestScore)

#%% Getting test score of polynomial regression
from sklearn.metrics import mean_squared_error
learnedModels = modelPipe.fit(transformedTrain, transformedLabels)
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = learnedModels.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)







