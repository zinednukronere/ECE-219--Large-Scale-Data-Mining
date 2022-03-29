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
from sklearn.ensemble import RandomForestRegressor
modelPipe = Pipeline([
    ('model',RandomForestRegressor(max_depth=500, max_features=5, n_estimators=150,oob_score=True))
    ])
#%% Getting the oob score
from sklearn.metrics import mean_squared_error
model = RandomForestRegressor(max_depth=500, max_features=5, n_estimators=150,oob_score=True)
trainedModel = model.fit(transformedTrain,transformedLabels)
oob_error = 1 - trainedModel.oob_score_

#%%
paramGrid = [
     {
          "model":[RandomForestRegressor()],
          "model__n_estimators":[50,100,150,200],
          "model__max_features":[3,5,7,9],
          "model__max_depth":[10,100,500,1000,2500,None]

     }
]
#%% Performing grid search over the parameters
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(modelPipe, paramGrid,cv=10,n_jobs=1, scoring='neg_mean_squared_error',
                    verbose=10,return_train_score=True)
grid.fit(transformedTrain, transformedLabels)
results = pd.DataFrame(grid.cv_results_)
print(grid.best_score_)
print(grid.best_params_)
#-0.018251480368939497
#{'model': RandomForestRegressor(max_depth=500, max_features=5, n_estimators=150), 
#'model__max_depth': 500, 'model__max_features': 5, 'model__n_estimators': 150}
#%%
import pickle

#with open('gridResultsRandomForest.pickle', 'wb') as handle:
#    pickle.dump(grid,handle)
#%%
import pickle
file = open("gridResultsRandomForest.pickle","rb")
grid2 = pickle.load(file)
results = pd.DataFrame(grid2.cv_results_)
#%% Gettingn test RMSE for otimized random forest
from sklearn.metrics import mean_squared_error
bestModel = grid.best_estimator_
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcessor.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featurePipe.transform(featuresTest)
preds = bestModel.predict(transformedTest)
rms = mean_squared_error(transformedTestLabels, preds, squared=False)
print(rms)
#%% Examining optmization results
from sklearn_evaluation import plot
plot.grid_search(grid2.cv_results_['mean_test_score'],change='model__max_depth', kind='bar')
#%% Examining optimization results
desiredfeats=['param_model__max_depth', 'param_model__max_features',
       'param_model__n_estimators','mean_test_score','mean_train_score']
k = results.loc[:,[feat for feat in desiredfeats]]
for feat in desiredfeats:
    k[feat]=pd.to_numeric(k[feat],errors = 'coerce')
k['diff']=abs(k.mean_train_score - k.mean_test_score)
#%% Plotting th auto correlation to see effect of features
import seaborn as sns
correlation_matrix = k.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', annot=True, cmap='Blues')
#%%
#TREE VISUALIZE
#%%
model = RandomForestRegressor(max_depth=4, max_features=5, n_estimators=150,oob_score=True)
trainedModel = model.fit(transformedTrain,transformedLabels)

#%%
depths = [estimator.tree_.max_depth for estimator in model.estimators_]
#%% Getting random tree ad plotting
from sklearn import tree
import random
featurenames = ['cut','color','clarity'] + numFeats
transformedDataFrame=pd.DataFrame(transformedTrain,columns=featurenames)
treeIndex = random.randrange(len(model.estimators_)+1)
#rf = RandomForestRegressor(n_estimators=50,max_depth=4)
#rf.fit(transformedDataFrame,transformedLabels)
plt.figure(figsize=(10,10))
_ = tree.plot_tree(model.estimators_[treeIndex], feature_names=transformedDataFrame.columns, filled=True,fontsize=10)













