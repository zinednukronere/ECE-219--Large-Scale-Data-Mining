#As grid search was done section by section, we need to combine the results
#%%
import pandas as pd
#%% Get different search results
fivefiftyresults=pd.read_csv("gridResults.csv")
fivehundredgaussian=pd.read_csv("gridResults500GaussianNB.csv")
fivehundredlogisticl1=pd.read_csv("gridResults500LogisticL1.csv")
fivehundredlogisticl2=pd.read_csv("gridResults500LogisticL2.csv")
fivehundredsvc=pd.read_csv("gridResults500SVC.csv")

#%% Combine the results
totalRes = pd.concat([fivefiftyresults, fivehundredgaussian,fivehundredlogisticl1,
           fivehundredlogisticl2,fivehundredsvc])

#%% 
totalRes.sort_values(by=['mean_test_score'],ascending=False)
