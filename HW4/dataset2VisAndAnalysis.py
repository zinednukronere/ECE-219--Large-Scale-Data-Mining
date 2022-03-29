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
labelProcesser = StandardScaler()
#%% Transform the data
featurenames = ['Year'] + numFeats
transformedFeatures = featureProcessPipe.fit_transform(featuresTrain)
labelsTrainProcessed = labelProcesser.fit_transform(np.array(labelsTrain).reshape(-1,1)).ravel()
transformedDataFrame=pd.DataFrame(transformedFeatures,columns=featurenames)
transformedDataFrame['CO'] = labelsTrainProcessed

#%% Draw the correlation matrix
import seaborn as sns
correlation_matrix = transformedDataFrame.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation_matrix, cbar=True, fmt='.1f', annot=True, cmap='Blues')
#%% Draw histograms of continous features
sns.histplot(x ="AT", data = transformedDataFrame)
plt.title("Histogram of AT")

plt.figure()
sns.histplot(x ="AP", data = transformedDataFrame)
plt.title("Histogram of AP")

plt.figure()
sns.histplot(x ="AH", data = transformedDataFrame)
plt.title("Histogram of AH")

plt.figure()
sns.histplot(x ="AFDP", data = transformedDataFrame)
plt.title("Histogram of AFDP")

plt.figure()
sns.histplot(x ="GTEP", data = transformedDataFrame)
plt.title("Histogram of GTEP")

plt.figure()
sns.histplot(x ="TIT", data = transformedDataFrame)
plt.title("Histogram of TIT")

plt.figure()
sns.histplot(x ="TAT", data = transformedDataFrame)
plt.title("Histogram of TAT")

plt.figure()
sns.histplot(x ="TEY", data = transformedDataFrame)
plt.title("Histogram of TEY")

plt.figure()
sns.histplot(x ="CDP", data = transformedDataFrame)
plt.title("Histogram of CDP")

#%%#%% Draw box plots of categorical features
sns.boxplot(x=featuresTrain.Year,y=labelsTrain, palette='rainbow',order=[2011,2012,2013,2014,2015])
#%%
#Yearly trends for each feature
#%%
years = [2011,2012,2013,2014,2015]
for year in years:
    dataYear = data.loc[data.Year==year,:]
    fig, axs = plt.subplots(3, 3,  sharex='col')

    axs[0,0].plot(dataYear.AT)
    axs[0,0].set(title="AT")
    axs[0,1].plot(dataYear.AP)
    axs[0,1].set(title="AP")
    axs[0,2].plot(dataYear.AH)
    axs[0,2].set(title="AH")
    axs[1,0].plot(dataYear.AFDP)
    axs[1,0].set(title="AFDP")
    axs[1,1].plot(dataYear.GTEP)
    axs[1,1].set(title="GTEP")
    axs[1,2].plot(dataYear.TIT)
    axs[1,2].set(title="TIT")
    axs[2,0].plot(dataYear.TAT)
    axs[2,0].set(title="TAT")
    axs[2,1].plot(dataYear.TEY)
    axs[2,1].set(title="TEY")
    axs[2,2].plot(dataYear.CDP)
    axs[2,2].set(title="CDP")
    title = 'Feature Plots for Year ' + str(year)
    fig.suptitle(title)

#%%
sns.lineplot(y="AT", data = data)
plt.figure()
sns.lineplot(x="Year", y="AP", data = data)
plt.figure()
sns.lineplot(x="Year", y="AH", data = data)
plt.figure()
sns.lineplot(x="Year", y="AFDP", data = data)
plt.figure()
sns.lineplot(x="Year", y="GTEP", data = data)
plt.figure()
sns.lineplot(x="Year", y="TIT", data = data)
plt.figure()
sns.lineplot(x="Year", y="TAT", data = data)
plt.figure()
sns.lineplot(x="Year", y="TEY", data = data)
plt.figure()
sns.lineplot(x="Year", y="CDP", data = data)
#%% Determine feature importance
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest

featureFilterFRegression = SelectKBest(f_regression, k='all')
newTrainFReg = featureFilterFRegression.fit_transform(transformedFeatures,labelsTrainProcessed)
print(featureFilterFRegression.scores_)

featureFilteMutInfo = SelectKBest(mutual_info_regression, k='all')
newTrainMutInf = featureFilteMutInfo.fit_transform(transformedFeatures,labelsTrainProcessed)
print(featureFilteMutInfo.scores_)
#%% Determining the effect of feature selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
featuresTest = test.loc[:,[feat for feat in list(test.columns) if feat!="CO"]]
labelsTest = test.CO
transformedTestLabels = labelProcesser.transform(np.array(labelsTest).reshape(-1,1))
transformedTest = featureProcessPipe.transform(featuresTest)
scoresList = []
for choose in range(1,11):
    featureFilterFRegression = SelectKBest(f_regression, k=choose)
    newTrainFReg = featureFilterFRegression.fit_transform(transformedFeatures,labelsTrainProcessed)
    rgr = LinearRegression()
    trainedLR= rgr.fit(newTrainFReg,labelsTrainProcessed)
    chosenTest = featureFilterFRegression.transform(transformedTest)
    preds = trainedLR.predict(chosenTest)
    rms = mean_squared_error(transformedTestLabels, preds, squared=False)
    print(rms)
    scoresList.append(np.mean(rms))

#%%
plt.plot(range(1,11),scoresList)
plt.title("Used Amount of Features vs Test RMSE")






