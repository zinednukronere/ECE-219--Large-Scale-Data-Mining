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
noScalefeatureProcessPipe = Pipeline([('A',preprocessColumns)])
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
#%% Defining linear regression model that will be fit
modelPipe = Pipeline([
    ('model',LinearRegression())
    ])
#%% Performing cross validation to see if pipeline good
from sklearn.model_selection import cross_validate
cv_results = cross_validate(modelPipe, transformedTrain, transformedLabels, cv=10, scoring='neg_mean_squared_error',verbose=1)
meanCVScore = np.mean(cv_results['test_score'])
print(meanCVScore)
#%% Getting test scores
from sklearn.metrics import mean_squared_error
trainedModelPipe = modelPipe.fit(transformedTrain, transformedLabels)
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = trainedModelPipe.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Performing search for regular lin regression, ridge and lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
paramGridLinear = [
     {
          "model":[LinearRegression()]
     },
     {
          "model":[Ridge()],
          "model__alpha":[0.01,0.1, 1.0, 10.0]
     },
     {
          "model":[Lasso()],
          "model__alpha":[0.01,0.1, 1.0, 10.0]
     }
]
#%% Determining best performance
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(modelPipe, paramGridLinear,cv=10,n_jobs=1, scoring='neg_mean_squared_error',verbose=10,
                    return_train_score=True)
grid.fit(transformedTrain, transformedLabels)
results = pd.DataFrame(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
#Best results with ridge regressio with alpha 10
#%% Using the best grid result to get test rmse
from sklearn.metrics import mean_squared_error
bestModel = grid.best_estimator_
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = bestModel.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Compare feature scaling for regular linear reg
from sklearn.metrics import mean_squared_error
withScalePipe = Pipeline([
    ('columnTransform',featureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',LinearRegression())
    ])
withoutScalePipe = Pipeline([
    ('columnTransform',noScalefeatureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',LinearRegression())
    ])

featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))

cv_results = cross_validate(withScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)
meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTestScore)
print(meanCVTrainScore)


trainedPipeScaled = withScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)


cv_results = cross_validate(withoutScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)
meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTestScore)
print(meanCVTrainScore)


trainedPipeNotScaled = withoutScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeNotScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)



#%% Compare feature scaling for ridge linear reg
from sklearn.metrics import mean_squared_error
withScalePipe = Pipeline([
    ('columnTransform',featureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',Ridge(alpha=100))
    ])
withoutScalePipe = Pipeline([
    ('columnTransform',noScalefeatureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',Ridge(alpha=100))
    ])

featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))

cv_results = cross_validate(withoutScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)

meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTrainScore)
print(meanCVTestScore)

trainedPipeNotScaled = withoutScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeNotScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)

cv_results = cross_validate(withScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)
meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTrainScore)
print(meanCVTestScore)


trainedPipeScaled = withScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)

#%% Compare feature scaling for lasso linear reg
from sklearn.metrics import mean_squared_error
withScalePipe = Pipeline([
    ('columnTransform',featureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',Lasso(alpha=0.1))
    ])
withoutScalePipe = Pipeline([
    ('columnTransform',noScalefeatureProcessPipe),
    ('feature_selection', SelectKBest(f_regression, k='all')),
    ('model',Lasso(alpha=0.1))
    ])

featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))


cv_results = cross_validate(withoutScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)
meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTrainScore)
print(meanCVTestScore)

trainedPipeNotScaled = withoutScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeNotScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)

cv_results = cross_validate(withScalePipe, featuresTrain, 
                            transformedLabels, cv=10, scoring='neg_mean_squared_error',
                            verbose=1,return_train_score=True)
meanCVTestScore = np.mean(cv_results['test_score'])
meanCVTrainScore = np.mean(cv_results['train_score'])
print(meanCVTrainScore)
print(meanCVTestScore)

trainedPipeScaled = withScalePipe.fit(featuresTrain, transformedLabels)
preds = trainedPipeScaled.predict(featuresTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Determining feature importance with p scores
import statsmodels.api as sm
featuresTrain = train.loc[:,[feat for feat in list(train.columns) if feat!="price"]]
labelsTrain = train.price
transformedLabels = labelProcessor.fit_transform(np.array(labelsTrain).reshape(-1,1)).ravel()
transformedTrain = featurePipe.fit_transform(featuresTrain,transformedLabels)
X = sm.add_constant(transformedTrain)
fit = sm.OLS(transformedLabels, X).fit()
for attributeIndex in range (0, transformedTrain.shape[1]):
    print(fit.pvalues[attributeIndex])


#%%
featurenames = ['cut','color','clarity'] + numFeats













