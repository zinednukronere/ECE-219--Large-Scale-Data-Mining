#%% Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Loading the dataframe
dataframe = pd.read_csv("Project_1_dataset_01_01_2022.csv")
numberOfRows,numberOfCols = dataframe.shape
#%% Couting the number of alpha numerial characters
numDigs = []
for row in range(0,numberOfRows):
    fullT = dataframe.iloc[row]["full_text"]
    numDigit = sum(c.isalnum() for c in fullT)
    numDigs.append(numDigit)
#%% Plotting the histogram of alpha numeric characters
plt.hist(numDigs,bins = 50)
plt.xlabel("Count of alpha-numeric characters")
plt.ylabel("Frequency")
plt.title("Histogram of alpha-numeric characters")
#%% Plotting the number of data samples for each category
plt.figure() 
dataframe['leaf_label'].value_counts().plot(kind='bar')
plt.ylabel("Frequency")
plt.figure() 
dataframe['root_label'].value_counts().plot(kind='bar')
plt.ylabel("Frequency")
#%% Importing libraries to divide data
import numpy as np
import random
np.random.seed(42)
random.seed(42)
#%% Spliting the data for training and testing
from sklearn.model_selection import train_test_split
from functions import clean
train, test = train_test_split(dataframe[["full_text","keywords","leaf_label", "root_label"]], test_size=0.2)
#%% Cleaning the data
trainSize = len(train)
testSize = len(test)
for i in range(0,trainSize):
    txt = train.iloc[i]["full_text"]
    train.iloc[i]["full_text"] = clean(txt)
for i in range(0,testSize):
    txt = test.iloc[i]["full_text"]
    test.iloc[i]["full_text"] = clean(txt)
#%% Definng the lemmatization and stemming functions
from nltk import pos_tag
#from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from sklearn.feature_extraction import text
from nltk.stem import PorterStemmer


stop_words = text.ENGLISH_STOP_WORDS
wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
tokenizer = CountVectorizer().build_analyzer()

def lemmatize_words(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag((text))]

def lemmaAndRemoveDigits(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    sentences = tokenize.sent_tokenize(text)
    tokenizer = CountVectorizer().build_analyzer()
    outputList = []
    for sentence in sentences:
        wordList = tokenizer(sentence)
        legalWords = [word for word in wordList if not any(ch.isdigit() for ch in word) and word not in stop_words ]
        output = [word for word in lemmatize_words(legalWords)]
        outputList = outputList + output
    return outputList

def stem_words(text):
    porter = PorterStemmer()
    return [porter.stem(word.lower()) 
            for word in (text)]

def stemAndRemoveDigits(text):
    sentences = tokenize.sent_tokenize(text)
    tokenizer = CountVectorizer().build_analyzer()
    outputList = []
    for sentence in sentences:
        wordList = tokenizer(sentence)
        legalWords = [word for word in wordList if not any(ch.isdigit() for ch in word) and word not in stop_words ]
        output = [word for word in stem_words(legalWords)]
        outputList = outputList + output
    return outputList

#%% Use count vectorizer and custom lemmatizzation function to analyze dataset
vectorize = CountVectorizer(min_df=3, 
                             analyzer=lemmaAndRemoveDigits,
                             stop_words='english')
trainCountVector = vectorize.fit_transform(train["full_text"])
testCountVector = vectorize.transform(test["full_text"])

#%% Examine shape  of count vectorizer output
print(trainCountVector.shape)
print(testCountVector.shape)
#%% Use tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTransformer = TfidfTransformer()
train_tfidf = tfidfTransformer.fit_transform(trainCountVector)
test_tfidf = tfidfTransformer.transform(testCountVector)
#%% Saving results so far for easier access
import pickle
with open('train_tfidf.pickle', 'wb') as handle:
    pickle.dump(train_tfidf,handle)
with open('test_tfidf.pickle', 'wb') as handle:
    pickle.dump(test_tfidf,handle)
#%% Loading the saved data to save time
import pickle
file = open("train_tfidf.pickle","rb")
train_tfidf = pickle.load(file)
file = open("test_tfidf.pickle","rb")
test_tfidf = pickle.load(file)
#%% Examine the tf-idf results
print(train_tfidf[0,:])
print(train_tfidf.shape)
print(test_tfidf.shape)
#%%  Dimesionality reduction using LSI 
from sklearn.decomposition import TruncatedSVD
lsi = TruncatedSVD(n_components=2000,random_state = 42)
lsiTrain = lsi.fit_transform(train_tfidf)
#%% Plot the explained ratio as number of components change
lsi = TruncatedSVD(n_components=2500,random_state = 42)
lsiTrain = lsi.fit_transform(train_tfidf)
componentNums = [1, 10, 50, 100, 200, 500, 1000, 2000]
explainedRatio = np.sort(lsi.explained_variance_ratio_)[::-1]
explainedVars = []
for cNum in componentNums:
    explainedVars.append(sum(explainedRatio[0:cNum]))

plt.plot(componentNums,explainedVars)  
plt.xlabel("Number of components")
plt.ylabel("Explained variance ratio")  
#%% LSI with 50 components
lsi = TruncatedSVD(n_components=50,random_state = 42)
lsiTrain = lsi.fit_transform(train_tfidf)
lsiTest = lsi.transform(test_tfidf)
#%% NMF with 50 components
from sklearn.decomposition import NMF
nmf = NMF(n_components=50, init='random', random_state=42)
nmfTrain = nmf.fit_transform(train_tfidf)
nmfTest = nmf.transform(test_tfidf)
#%% Calculating frobenius norm for LSI with 50 components 
from sklearn.utils.extmath import randomized_svd
u,sigma,vt = randomized_svd(train_tfidf,n_components=50)
x_50 = np.dot(np.dot(u,np.diag(sigma)),vt)
lsi_frobenius = np.sqrt(np.sum(np.square(x_50-train_tfidf)))
print("Frobenius Norm for LSI: \n" + str(lsi_frobenius))

#%%Frob Norm for NMF with 50 components
H = nmf.components_
nmfFrob = np.sqrt(np.sum(np.array(train_tfidf - nmfTrain.dot(H))**2))
print("Frobenius Norm for NMF: \n" + str(nmfFrob))
#%% Defining the data and labels for model fitting
toTrain = lsiTrain
trainLabels = train["root_label"]
toTest = lsiTest
testLabels = test["root_label"]

#%% Fitting linear SVM with C=1000
from sklearn.svm import SVC
from functions import calculatePerfParams
svm1 = SVC(kernel = "linear",C = 1000,probability=False)
svm1.fit(toTrain,trainLabels)
predicted = svm1.predict(toTest)
acc,prec,recall,f1 = calculatePerfParams(svm1,toTest,testLabels,"climate")

#%% Fitting linear SVM with C=0.0001
svm2 = SVC(kernel = "linear",C = 0.0001,probability=False)
svm2.fit(toTrain,trainLabels)
predicted = svm2.predict(toTest)
acc2,prec2,recall2,f12 = calculatePerfParams(svm2,toTest,testLabels,"climate")
#%% Fitting linear SVM with C=100000
svm3 = SVC(kernel = "linear",C = 100000)
svm3.fit(toTrain,trainLabels)
predicted = svm3.predict(toTest)
acc3,prec3,recall3,f13 = calculatePerfParams(svm3,toTest,testLabels,"climate")
#%% Cross validation for best C parameter
from sklearn.model_selection import cross_val_score
gammaValues = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5,1e6]
best_score = 0
for gamma in gammaValues:
    svm = SVC(kernel='linear', C=gamma)
    scores = cross_val_score(svm, toTrain, trainLabels, cv=5, scoring='accuracy')
    print('gamma = ', gamma, ' score = ', np.mean(scores))
    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_gamma = gamma
        
#Best gamma is found to be 1000
#%% Training svm with best C
svmBest = SVC(kernel = "linear",C = best_gamma)
svmBest.fit(toTrain,trainLabels)
predicted = svmBest.predict(toTest)
accB,precB,recallB,f1B = calculatePerfParams(svmBest,toTest,testLabels,"climate")

#%% Logistic Regression without penalty
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(penalty="none",max_iter=500)
logReg.fit(toTrain,trainLabels)
accB,precB,recallB,f1B = calculatePerfParams(logReg,toTest,testLabels,"climate")
#%% Determining the best penalty for L1 logistic regression
penaltyValues = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
best_score = 0
for penaltyV in penaltyValues:
    lReg = LogisticRegression(penalty='l1', C=penaltyV, solver = "liblinear",max_iter=500)
    scores = cross_val_score(lReg, toTrain, trainLabels, cv=5, scoring='accuracy')
    print('penalty = ', penaltyV, ' score = ', np.mean(scores))
    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_penalty_l1 = penaltyV
#best penalty found as 100
#%% Determining the best penalty for L2 logistic regression
penaltyValues = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
best_score = 0
for penalty in penaltyValues:
    lReg = LogisticRegression(penalty='l2', C=penalty, solver = "liblinear",max_iter=500)
    scores = cross_val_score(lReg, toTrain, trainLabels, cv=5, scoring='accuracy')
    print('penalty = ', penalty, ' score = ', np.mean(scores))
    if np.mean(scores) > best_score:
        best_score = np.mean(scores)
        best_penalty_l2 = penalty
#best penalty found as 1e4
#%% Training the logistic regression models ith varyin penalties
logRegNoPenalty = LogisticRegression(penalty="none",max_iter=500)
logRegNoPenalty.fit(toTrain,trainLabels)
accNP,precNP,recallNP,f1NP = calculatePerfParams(logRegNoPenalty,toTest,testLabels,"climate")

logRegL1Penalty = LogisticRegression(penalty='l1', C=best_penalty_l1, solver = "liblinear",max_iter=500)
logRegL1Penalty.fit(toTrain,trainLabels)
accLP,precLP,recallLP,f1LP = calculatePerfParams(logRegL1Penalty,toTest,testLabels,"climate")

logRegL2Penalty = LogisticRegression(penalty='l2', C=best_penalty_l2, solver = "liblinear",max_iter=500)
logRegL2Penalty.fit(toTrain,trainLabels)
accL2,precL2,recallL2,f1L2 = calculatePerfParams(logRegL2Penalty,toTest,testLabels,"climate")

#%% Fitting gaussian NB model
from sklearn.naive_bayes import GaussianNB

gaussianNB = GaussianNB()
gaussianNB.fit(toTrain,trainLabels)
accNB,precNB,recallNB,f1NB = calculatePerfParams(gaussianNB,toTest,testLabels,"climate")

#%% Defining data and labels to use for Multiclass Classification
from functions import calculateMulticlassPerfParams
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

toTrain = lsiTrain
trainLabels = train["leaf_label"]
toTest = lsiTest
testLabels = test["leaf_label"]
labels = ["chess","cricket","soccer","football","%22forest%20fire%22","flood","earthquake","drought"]

#%% Performing multiclass Naive Bayes Classification
gaussianMultiNB = GaussianNB()
gaussianMultiNB.fit(toTrain,trainLabels)
acc,prec,recall,f1 = calculateMulticlassPerfParams(gaussianMultiNB, toTest, testLabels,labels=labels)

#%%Performing One vs. One SVM Classification
svmOVO = OneVsOneClassifier(SVC(kernel = "linear",C = 1000))
svmOVO.fit(toTrain,trainLabels)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVO, toTest, testLabels,labels=labels)

#%% Performing One vs. Rest SVM Classification

# The “balanced” mode uses the values of y to automatically adjust weights 
# inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
svmOVR = OneVsRestClassifier(SVC(kernel = "linear",C = 1000, class_weight='balanced'))
svmOVR.fit(toTrain,trainLabels)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVR, toTest, testLabels,labels=labels)

#%% merge flood & earthquake, maybe drought & forestfire
trainLabelsMerged = pd.Series.copy(trainLabels)
testLabelsMerged = pd.Series.copy(testLabels)
trainLabelsMerged[trainLabelsMerged == "earthquake"] = "merged"
trainLabelsMerged[trainLabelsMerged == "flood"] = "merged"

testLabelsMerged[testLabelsMerged == "earthquake"] = "merged"
testLabelsMerged[testLabelsMerged == "flood"] = "merged"

labels = ["chess","cricket","soccer","football","%22forest%20fire%22","merged","drought"]

gaussianMultiNB = GaussianNB()
gaussianMultiNB.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(gaussianMultiNB, toTest, testLabelsMerged,labels=labels)

svmOVO = OneVsOneClassifier(SVC(kernel = "linear",C = 1000))
svmOVO.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVO, toTest, testLabelsMerged,labels=labels)

svmOVR = OneVsRestClassifier(SVC(kernel = "linear",C = 1000, class_weight=None))
svmOVR.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVR, toTest, testLabelsMerged,labels=labels)

#%% fix class imbalance for merged classes
gaussianMultiNB = GaussianNB()
gaussianMultiNB.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(gaussianMultiNB, toTest, testLabelsMerged,labels=labels)

svmOVO = OneVsOneClassifier(SVC(kernel = "linear",C = 1000, class_weight='balanced'))
svmOVO.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVO, toTest, testLabelsMerged,labels=labels)

svmOVR = OneVsRestClassifier(SVC(kernel = "linear",C = 1000, class_weight='balanced'))
svmOVR.fit(toTrain,trainLabelsMerged)
acc,prec,recall,f1 = calculateMulticlassPerfParams(svmOVR, toTest, testLabelsMerged,labels=labels)

#%% 10 - GLoVE Embedding Questions
# a)    The ratio is better able to distinguish between relevant and irrelevant words
#       and is also better able to discriminate between two relevant words.
# b)    They would return the same vector because the embeddings are pretrained
#       and are not context sensitive.
# c)    |GLoVE['queen'] - GLoVE['king'] - GLoVE['wife'] + GLoVE['husband']| = 0
#       |GLoVE['queen'] - GLoVE['king']| = |GLoVE['wife'] - GLoVE['husband']|
# d)    It would be more appropriate to lemmatize rather than stem since, stemming produces some non-words

#%% Loading GLoVE Embeddings
embeddings_dict = {}
dimension_of_glove = 300
with open("glove/glove.6B.300d.txt", 'r',encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
#%%
lenFullTs = []
for row in range(0,numberOfRows):
    fullT = dataframe.iloc[row]["full_text"]
    lenFullTs.append(len(fullT.replace(" ", "")))
#%% 
plt.hist(lenFullTs,bins = 1000)
#%% 
import ast
lenKeywords = []
for row in range(0,numberOfRows):
    keywords = ast.literal_eval(dataframe.iloc[row]["keywords"])
    lenKeywords.append(len(keywords))
plt.hist(lenKeywords,bins = max(lenKeywords))
#%% GLoVE Feature Engineering
import ast
from functions import featurizeKeywords
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

trainFeature = featurizeKeywords(train, embeddings_dict,dimension_of_glove)
testFeature = featurizeKeywords(test, embeddings_dict,dimension_of_glove)
#%% Define the labels used for binary classification
trainLabels = train["root_label"]
testLabels = test["root_label"]

#%% 11 - Train a Classifier
classifier = LogisticRegression(penalty='l2', solver = "liblinear",max_iter=500)
classifier.fit(trainFeature,trainLabels)
accGLoVE,precGLoVE,recallGLoVE,f1GLoVE = calculatePerfParams(classifier,testFeature,testLabels,"sports")

#%% 12 - GLoVE Dimension analysis
glove_accs = []
for glove_fname, dimension_of_glove  in zip(["glove.6B.50d.txt","glove.6B.100d.txt","glove.6B.200d.txt","glove.6B.300d.txt"],[50,100,200,300]):
    embeddings_dict = {}
    with open("glove/" + glove_fname, 'r',encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    trainFeature = featurizeKeywords(train, embeddings_dict,dimension_of_glove)
    testFeature = featurizeKeywords(test, embeddings_dict,dimension_of_glove)
    classifier = LogisticRegression(penalty='l2', solver = "liblinear",max_iter=500)
    classifier.fit(trainFeature,trainLabels)
    accGLoVE,precGLoVE,recallGLoVE,f1GLoVE = calculatePerfParams(classifier,testFeature,testLabels,"sports")
    glove_accs+=[accGLoVE]
#%%
plt.figure()
plt.plot([50,100,200,300], glove_accs,"b*--")
plt.xlabel("GLoVE length")
plt.ylabel("Testing accuracy (%)")

#%% 13 - UMAP Visualization
import umap.umap_ as umap
import seaborn as sns

# generate UMAP embeddings
reducer = umap.UMAP()
rand_uniform = StandardScaler().fit_transform(np.random.uniform(size=trainFeature.shape))
scaled_data = StandardScaler().fit_transform(trainFeature)
UMAP_GLOVE_embedding = reducer.fit_transform(scaled_data)
UMAP_rand_embedding = reducer.fit_transform(rand_uniform)

#%% Plotting glove based embeddings
plt.figure(figsize=(10,5))
x=0
for label in trainLabels.unique():
    indices = [i for i in range(0,len(trainLabels)) if trainLabels.iloc[i]==label]
    plt.scatter(
    UMAP_GLOVE_embedding[indices, 0],
    UMAP_GLOVE_embedding[indices, 1],
    s=5,
    c=sns.color_palette()[x],
    label=label)
    x=1
    print(label)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the GLoVE Embeddings', fontsize=12)
plt.legend()

#%% Plotting of random vectors
plt.figure(figsize=(10,5))
x=0
for label in trainLabels.unique():
    indices = [i for i in range(0,len(trainLabels)) if trainLabels.iloc[i]==label]
    plt.scatter(
    UMAP_rand_embedding[indices, 0],
    UMAP_rand_embedding[indices, 1],
    s=5,
    c=sns.color_palette()[x],label=label)
    x=1
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Random Embeddings', fontsize=12)
plt.legend()













