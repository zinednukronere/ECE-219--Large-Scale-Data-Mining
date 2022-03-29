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
labelProcesser = StandardScaler()
#%% Transform the data
featurenames = ['cut','color','clarity'] + numFeats
transformedFeatures = featureProcessPipe.fit_transform(featuresTrain)
labelsTrainProcessed = labelProcesser.fit_transform(np.array(train.price).reshape(-1,1)).ravel()
transformedDataFrame=pd.DataFrame(transformedFeatures,columns=featurenames)
transformedDataFrame['price'] = labelsTrainProcessed
#%% Draw the correlation matrix
import seaborn as sns
correlation_matrix = transformedDataFrame.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', annot=True, cmap='Blues')

#%% Draw histograms of continous features
sns.histplot(x ="carat", data = transformedDataFrame)
plt.title("Histogram of carat")

plt.figure()
sns.histplot(x ="x", data = transformedDataFrame)
plt.title("Histogram of x")

plt.figure()
sns.histplot(x ="y", data = transformedDataFrame)
plt.title("Histogram of y")

plt.figure()
sns.histplot(x ="z", data = transformedDataFrame)
plt.title("Histogram of z")

plt.figure()
sns.histplot(x ="depth", data = transformedDataFrame)
plt.title("Histogram of depth")

plt.figure()
sns.histplot(x ="table", data = transformedDataFrame)
plt.title("Histogram of table")

#%% Draw box plots of categorical features
sns.boxplot(x=featuresTrain.cut,y=labelsTrain, palette='rainbow',order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
plt.title("Box plot of cut")
plt.ylabel("Price")

plt.figure()
sns.boxplot(x=featuresTrain.color,y=labelsTrain, palette='rainbow',order=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
plt.title("Box plot of color")
plt.ylabel("Price")


plt.figure()
sns.boxplot(x=featuresTrain.clarity,y=labelsTrain, palette='rainbow',order=['I1', 'SI2', 'SI1', 'VS2','VS1', 'VVS2', 
                                           'VVS1', 'IF'])
plt.title("Box plot of clarity")
plt.ylabel("Price")

#%% Draw count plots of categorical features
sns.catplot(x ="color",kind ="count", data = featuresTrain,order=['J', 'I', 'H', 'G', 'F', 'E', 'D'])

sns.catplot(x ="cut",kind ="count", data = featuresTrain,order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])

sns.catplot(x ="clarity",kind ="count", data = featuresTrain,order=['I1', 'SI2', 'SI1', 'VS2','VS1', 'VVS2', 
                                           'VVS1', 'IF'])
#%% Determine feature importance
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

featureFilterFRegression = SelectKBest(f_regression, k=4)
newTrainFReg = featureFilterFRegression.fit_transform(transformedFeatures,labelsTrainProcessed)
print(featureFilterFRegression.scores_)

featureFilteMutInfo = SelectKBest(mutual_info_regression, k=5)
newTrainMutInf = featureFilteMutInfo.fit_transform(transformedFeatures,labelsTrainProcessed)
print(featureFilteMutInfo.scores_)
#%% Determining the effect of feature selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="price"]]
labelsTest = test.price
transformedTestLabels = labelProcesser.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featureProcessPipe.transform(featuresTest)
scoresList = []
for choose in range(1,10):
    featureFilterFRegression = SelectKBest(mutual_info_regression, k=choose)
    newTrainFReg = featureFilterFRegression.fit_transform(transformedFeatures,labelsTrainProcessed)
    rgr = LinearRegression()
    trainedLR= rgr.fit(newTrainFReg,labelsTrainProcessed)
    chosenTest = featureFilterFRegression.transform(transformedTest)
    preds = trainedLR.predict(chosenTest)
    rms = mean_squared_error(transformedTestLabels, preds, squared=False)
    print(rms)
    scoresList.append(np.mean(rms))
    
#scores2 = cross_val_score(rgr, newTrainFReg, labelsTrainProcessed,scoring='neg_mean_squared_error', cv=5)
#print(np.mean(scores2))

#Using all the features proved to be more effective

#%%
plt.plot(range(1,10),scoresList)
plt.title("Used Amount of Features vs Test RMSE")









