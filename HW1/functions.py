import re
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk import pos_tag
#from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem import PorterStemmer
from nltk import tokenize
from sklearn.feature_extraction import text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import ast

#Remove html tags
def clean(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    texter = re.sub(r"<br />", " ", text)
    texter = re.sub(r"&quot;", "\"",texter)
    texter = re.sub('&#39;', "\"", texter)
    texter = re.sub('\n', " ", texter)
    texter = re.sub(' u '," you ", texter)
    texter = re.sub('`',"", texter)
    texter = re.sub(' +', ' ', texter)
    texter = re.sub(r"(!)\1+", r"!", texter)
    texter = re.sub(r"(\?)\1+", r"?", texter)
    texter = re.sub('&amp;', 'and', texter)
    texter = re.sub('\r', ' ',texter)
    clean = re.compile('<.*?>')
    texter = texter.encode('ascii', 'ignore').decode('ascii')
    texter = re.sub(clean, '', texter)
    if texter == "":
        texter = ""
    return texter

#Determine metrics, plot roc and conf matrix for binary classification
def calculatePerfParams(model,testData,testLabels,positiveLabel):
    predictions = model.predict(testData)
    acc = metrics.accuracy_score(testLabels, predictions)
    prec = metrics.precision_score(testLabels, predictions,pos_label = positiveLabel)
    recall = metrics.recall_score(testLabels, predictions,pos_label = positiveLabel)
    f1 = metrics.f1_score(testLabels, predictions,pos_label = positiveLabel)
    
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(prec))
    print('Recall: ' + str(recall))
    print('F-1 Score: ' + str(f1))
    
    #plt.figure()
    metrics.plot_confusion_matrix(model, testData, testLabels) 
    
    
    #plt.figure()
    metrics.plot_roc_curve(model, testData, testLabels,pos_label = positiveLabel) 
    
    return acc,prec,recall,f1

#Used to determine whether data should be cleaned in pipeline
class Cleaner(TransformerMixin, BaseEstimator):
    def __init__(self,toClean):
        self.toClean = toClean
    def cleanDataFullText(self,data):
        newData = data.copy()
        size = len(newData)
        for i in range(0,size):
            txt = newData.iloc[i]
            newData.iloc[i] = clean(txt)
        return newData
    def fit(self, x, y=None):
        return self
    def transform(self, X):
        if self.toClean:
            return self.cleanDataFullText(X)
        else:
            return X



def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_words(text): 
    # Text input is string, returns lowercased strings.
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag((text))]

def stem_words(text):
    porter = PorterStemmer()
    return [porter.stem(word.lower()) 
            for word in (text)]

#Checks if word has digits or stop word. If not, lemmatizes and outputs
def lemmaAndRemoveDigits(textIn):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    stop_words = text.ENGLISH_STOP_WORDS
    sentences = tokenize.sent_tokenize(textIn)
    tokenizer = CountVectorizer().build_analyzer()
    outputList = []
    for sentence in sentences:
        wordList = tokenizer(sentence)
        legalWords = [word for word in wordList if not any(ch.isdigit() for ch in word) and word not in stop_words ]
        output = [word for word in lemmatize_words(legalWords)]
        outputList = outputList + output
    return outputList

#Checks if word has digits or stop word. If not, outputs
def nothingAndRemoveDigits(textIn):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    tokenizer = CountVectorizer().build_analyzer()
    stop_words = text.ENGLISH_STOP_WORDS
    return [word for word in (tokenizer(textIn)) 
            if not any(ch.isdigit() for ch in word) and word not in stop_words]

#Checks if word has digits or stop word. If not, stems
def stemAndRemoveDigits(textIn):
    stop_words = text.ENGLISH_STOP_WORDS
    sentences = tokenize.sent_tokenize(textIn)
    tokenizer = CountVectorizer().build_analyzer()
    outputList = []
    for sentence in sentences:
        wordList = tokenizer(sentence)
        legalWords = [word for word in wordList if not any(ch.isdigit() for ch in word) and word not in stop_words ]
        output = [word for word in stem_words(legalWords)]
        outputList = outputList + output
    return outputList

#Creates the features for the keywords of a data sample using glove embeddings
def featurizeKeywords(df, embeddings, glove_dim):
    """Takes in a dataframe and embeddings and converts them to an array the same 
    length x 300 which uses the GLoVE Embeddings"""
    
    features = np.empty((0,glove_dim), 'float32')
    for row in range(0,len(df)):
        keywords = ast.literal_eval(df.iloc[row]["keywords"])
       # keywords = lemmatize_words(keywords)
        feature = np.zeros(glove_dim)
        for keyword in keywords:
            if keyword in embeddings:
                feature += embeddings[keyword]
        features = np.vstack((features, feature))
    return features
 
#Calculate metrics and plot confusion matrix for multiclass classification
def calculateMulticlassPerfParams(model,testData,testLabels,labels):
    predictions = model.predict(testData)
    acc = metrics.accuracy_score(testLabels, predictions)
    prec = metrics.precision_score(testLabels, predictions, average='macro' )
    recall = metrics.recall_score(testLabels, predictions, average='macro')
    f1 = metrics.f1_score(testLabels, predictions, average='macro')
    
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(prec))
    print('Recall: ' + str(recall))
    print('F-1 Score: ' + str(f1))
    
    #plt.figure()
    cm = metrics.confusion_matrix(testLabels, predictions,labels = labels) 
    # plt.imshow(cm)
    # plt.colorbar()
    # plt.plot()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical')
    plt.show()

    # metrics.plot_roc_curve(model, testData, testLabels) 
    
    return acc,prec,recall,f1