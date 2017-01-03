#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on Fri Nov 25 23:27:00 2016

@author: leena
"""

# managing all the import for use in the code
import re
import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
import sys

# regex to remove unwanted words/symbols from text
_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'r'|^In article|^Quoted from|^\||^>)')
stemmer = PorterStemmer()
lemmetizer = WordNetLemmatizer()
                  
def main():
    
    
    if len(sys.argv) > 1:
        testDirPath = sys.argv[1]
        #testDirPath = './Selected20NewsGroup/Test'
    else:
        print('Format to run this program: "python hw3_classification.py <path_to_test_data>"')
    try:

        mbc_model = load_model('my_classifier.joblib.pkl')
        #if these parameters are set then header and other things are removed else not.
        test_data, test_target, target_names = get_stripped_data(testDirPath, True, True, True)
        mbc_classifier = mbc_model[0]
        mbc_vectorizer = mbc_model[1]
        X_test = mbc_vectorizer.transform(test_data)
        prediction = mbc_classifier.predict(X_test)
        
        print("Classification report:")
        print(metrics.classification_report(test_target, prediction, target_names=target_names))
        
    except:
        print('Something went wrong while loading pre trained model; Training the model')
        start_model_training()
    
    
    
def start_model_training():
    
    stripping = False
    stemming = False
    lemmetizing = False
    
    train_data, train_target, train_target_names = get_stripped_data('./Selected20NewsGroup/Training', stripping, stemming, lemmetizing)
    test_data, test_target, test_target_names = get_stripped_data('./Selected20NewsGroup/Test', stripping, stemming, lemmetizing)
    my_best_model(train_data, train_target, test_data, test_target, test_target_names)     

#    NB_classifier = MultinomialNB(alpha=0.01)
#    LR_classifier = linear_model.SGDClassifier(loss='log', alpha=0.0001, n_iter=50, penalty='l2')
#    SVM_classifier = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, n_iter=50)
#    RF_classifier = RandomForestClassifier(n_estimators=50, max_features=500)
#    
#    
#    print("\n\n********** Unigram Models **********")
#    unigram_tf_model(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names)
#    
#    print("\n\n********** Bigram Models **********")
#    bigram_tf_model(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names)
    
#    chi_sq_kernel(train_data, test_data, train_target, test_target, test_target_names) #30.428


############################## My best model ############################## 

    
def my_best_model(train_data, train_target, test_data, test_target, target_names):
    
    print("\n\n********** Inside My best model **********") 
    
    SVM_classifier = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, n_iter=50)
    vectorizer = TfidfVectorizer(tokenizer= Tokenizer(), ngram_range=(1,1), sublinear_tf=True, use_idf=True, stop_words='english', lowercase=True, norm= 'l2')
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    #X_train, X_test = get_chi_sq_features(X_train, X_test, train_target, vectorizer)
    train_classifier(SVM_classifier, X_train, X_test, train_target, test_target, vectorizer, target_names)
    
    save_model(SVM_classifier, vectorizer)
    
    #SVC_classifier = SVC(C=1000000.0, gamma=1.0000000000000001e-05)
    #train_classifier(SVC_classifier, X_train, X_test, train_target, test_target, vectorizer, target_names)


       
############################## Grid search ############################## 

def do_grid_search(train_data, train_target):
    
    NB_grid_search(train_data, train_target) 
    LR_grid_search(train_data, train_target)
    SVM_grid_search(train_data, train_target)
    RF_grid_search(train_data, train_target)

    
############################## UNigram and Bigram models ############################## 
    

def unigram_tf_model(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names):
    
    uni_vectorizer = TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, stop_words='english', lowercase=True, use_idf=True)
    X_train = uni_vectorizer.fit_transform(train_data)
    X_test = uni_vectorizer.transform(test_data)
    train_classifiers_and_generate_reports(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, X_train, X_test, train_target, test_target, uni_vectorizer, test_target_names)
    
    #generate learning curve for all the models
    classifiers = []
    classifiers.extend((NB_classifier, LR_classifier, SVM_classifier, RF_classifier))
    custom_learning_curve(classifiers, X_train, train_target, X_test, test_target)
    
    
def bigram_tf_model(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names):
    
    bi_vectorizer = TfidfVectorizer(ngram_range=(2,2), sublinear_tf=True, stop_words='english', lowercase=True, use_idf=True)
    X_train = bi_vectorizer.fit_transform(train_data)
    X_test = bi_vectorizer.transform(test_data)
    train_classifiers_and_generate_reports(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, X_train, X_test, train_target, test_target, bi_vectorizer, test_target_names)

def unigram_count_model(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names):
    
    uni_count_vectorizer = CountVectorizer(ngram_range=(1,1), stop_words='english', lowercase=True)
    X_train = uni_count_vectorizer.fit_transform(train_data)
    X_test = uni_count_vectorizer.transform(test_data)    
    train_classifiers_and_generate_reports(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, X_train, X_test, train_target, test_target, uni_count_vectorizer, test_target_names)

    
def bigra_count_models(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, train_data, train_target, test_data, test_target, test_target_names):
    
    bi_count_vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english', lowercase=True)
    X_train = bi_count_vectorizer.fit_transform(train_data)
    X_test = bi_count_vectorizer.transform(test_data)
    train_classifiers_and_generate_reports(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, X_train, X_test, train_target, test_target, bi_count_vectorizer, test_target_names)

    
def train_classifiers_and_generate_reports(NB_classifier, LR_classifier, SVM_classifier, RF_classifier, X_train, X_test, train_target, test_target, vectorizer, test_target_names):
    
    train_classifier(NB_classifier, X_train, X_test, train_target, test_target, vectorizer, test_target_names)
    train_classifier(LR_classifier, X_train, X_test, train_target, test_target, vectorizer, test_target_names)
    train_classifier(SVM_classifier, X_train, X_test, train_target, test_target, vectorizer, test_target_names)
    train_classifier(RF_classifier, X_train, X_test, train_target, test_target, vectorizer, test_target_names)
    
        
############################## Feature extraction ############################## 

#reading data from disk and removing header, footer, quotes, punctuations 
def get_stripped_data(path_to_files, stripping, stemming, lemmetizing):
    
    bunch = load_files(path_to_files, encoding='latin1') 

    data = bunch.data
    target = bunch.target
    target_names = bunch.target_names
    
    filenames = bunch.filenames
    print("Loaded: ", len(filenames), "files from", path_to_files) 

    #training data 
    if stripping:
        data = [strip_newsgroup_header(text) for text in data]
        data = [strip_newsgroup_footer(text) for text in data]
        data = [strip_newsgroup_quoting(text, stemming, lemmetizing) for text in data]
                
    return data, target, target_names    

#remove header
def strip_newsgroup_header(text):
    
    _before, _blankline, after = text.partition('\n\n')
    return after

#remove quotes
def strip_newsgroup_quoting(text, stemming, lemmetizing):
    
    if stemming or lemmetizing:
        return strip_and_preprocess_newsgroup_quoting(text, stemming, lemmetizing)
    else:
        good_lines = [line for line in text.split('\n') if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)
    
    
###remove quotes, do stemming, do lemmetizing    
def strip_and_preprocess_newsgroup_quoting(text, stemming, lemmetizing):
    
    lines = []
    each_line = ''
    for line in text.split('\n'):
        if not line == '' or line:
            if not _QUOTE_RE.search(line):
                tokens = word_tokenize(line)
                tokens = [i for i in tokens if i not in string.punctuation]
                if lemmetizing:
                    tokens = [lemmetizer.lemmatize(t) for t in tokens]
                each_line= ' '.join(stemmer.stem(word) for word in tokens)
                lines.append(each_line)
    
    return '\n'.join(lines)

#remove footer
def strip_newsgroup_footer(text):
    
    lines = text.split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

############################## Feature selection ############################## 

#1. remove features that are in more than given value of the dataset       
def remove_low_variance_features(X_train, value):
    
    threshold=(value * (1 - value))
    selection = VarianceThreshold(threshold)
    selection.fit(X_train)


#choosing best features based on Chi sq test
def get_chi_sq_features(X_train, X_test, Y_train, vectorizer):
    
    ch2 = SelectKBest(chi2, 1000)
    X_train = ch2.fit_transform(X_train, Y_train)
    X_test = ch2.transform(X_test)
    feature_names = vectorizer.get_feature_names()

    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    if feature_names:
        feature_names = np.asarray(feature_names)
    
    #print(feature_names)
    return X_train, X_test
             
def chi_sq_kernel(train_data, test_data, train_target, test_target, target_names):
    
    vectorizer = TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True, use_idf=True, stop_words='english', lowercase=True, norm= 'l2')
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    SVM_classifier = SVC(kernel=chi2_kernel).fit(X_train.toarray(), train_target=target_names)
    prediction = SVM_classifier.predict(X_test.toarray())

    print("Classification report:")
    print(metrics.classification_report(test_target, prediction))     
    
    
############################## Model persistence and load check ############################## 

def load_model(filename):
    
    print('Loading pre-trained model')
    if not filename:
        filename = 'my_classifier.joblib.pkl'
    loaded_model = joblib.load(filename) 
    return loaded_model

    
def save_model(model, vectorizer):
    
    print('Saving model to disk')
    filename = 'my_classifier.joblib.pkl'
    _ = joblib.dump([model, vectorizer], filename)
    print(_)



############################## Training classifiers and reporting scores ############################## 
    
# function that implements training of a classifier   
def train_classifier(classifier, X_train, X_test, Y_train, Y_test, vectorizer, target_names):
    
    print("\n********** Training a new classifier **********")
    #print(classifier)
    
    classifier.fit(X_train, Y_train)
    prediction = classifier.predict(X_test)

    print("Classification report:")
    print(metrics.classification_report(Y_test, prediction, target_names=target_names))

    #cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    #plot_learning_curve(classifier, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    
def get_scores_for_plot(classifier, X_train, Y_train, X_test, Y_test):
    
    scores = []
    sizes = []
    init_size = 155
    for x in range(1, 15):
        length = x * init_size
        sizes.append(length)
        
        #shuffling both the training data and target 
        X_train, Y_train = shuffle(X_train, Y_train)
        sliced_train_data = X_train[1:length]
        sliced_train_target = Y_train[1:length]
        classifier.fit(sliced_train_data, sliced_train_target)
        
        prediction = classifier.predict(X_test)
        
        f1_score = metrics.f1_score(Y_test, prediction, average='macro')
        scores.append(f1_score)
     
    return sizes, scores    


############################## Plotting graphs ############################## 

#function to plot f1 scores vs Training size for all the classifiers
def custom_learning_curve(classifiers, X_train, Y_train, X_test, Y_test):

    color = ['r', 'g', 'b', 'm']

    for x in range(len(classifiers)):
        
        try:
            identifier = classifiers[x].loss
        except:
            identifier = ''
        sizes, scores = get_scores_for_plot(classifiers[x], X_train, Y_train, X_test, Y_test)
        plt.plot(sizes, scores, marker='o', color=color[x], label=str(classifiers[x]).split('(')[0]+ '-' +identifier)
        plt.legend(loc=4)
#        for xy in zip(sizes, scores):                                       
#        pyplot.annotate('(%s, %.3f)' % xy, xy=xy, textcoords='data')
    
    plt.ylabel('F1-score')
    plt.xlabel('Training Size')
    plt.grid()
    plt.show()
    plt.savefig('plots/F1-scoreVsTrainingSize')
    plt.clf()

    
# function to plot Learning curve for each classifier on training data using cross validation
def plot_learning_curve(classifier, X, Y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    #plot data
    train_sizes, train_scores, test_scores = learning_curve(classifier, X, Y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    #plot properties
    
    title = "LearningCurve " + (str(classifier).split('(')[0])
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("Training size")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    plt.savefig('plots/%s'%title)
    plt.clf()


    
############################## Grid search to find best parameters for models ############################## 

#Steps- Build a pipeline, Define parameters, Fit model, Fit model, FInd best parameters
def NB_grid_search(train_data, train_target):
    
    NB_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
    NB_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
    find_best_paramas_for_model(NB_classifier, NB_parameters, train_data, train_target)
   
    
def SVM_grid_search(train_data, train_target):
    
    SVM_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC()),])    
    SVM_parameters = {'clf__kernel': ['rbf', 'poly', 'linear'], 'clf__C': np.logspace(-2, 10, 13), 'clf__gamma': np.logspace(-9, 3, 13) }
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    gs_classifier = GridSearchCV(SVM_classifier, SVM_parameters, cv=cv)
    gs_classifier.fit(train_data, train_target)
    
    print('Best score: ', gs_classifier.best_score_)
    for param_name in sorted(SVM_parameters.keys()):
        print("%s: %r" % (param_name, gs_classifier.best_params_[param_name])) 
        

def LR_grid_search(train_data, train_target):
    
    LR_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', linear_model.SGDClassifier(loss='log')),])
    LR_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (0.0001, 0.001, 0.01, 0.1, 1, 10), 'clf__penalty': ('l1', 'l2', 'elasticnet'), 'clf__n_iter': (10, 50, 80, 100)}
    find_best_paramas_for_model(LR_classifier, LR_parameters, train_data, train_target)
    
        
def RF_grid_search(train_data, train_target):
    
    RF_classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier())])
    parameters = {'vect__ngram_range': ((1, 1), (1,2)), 'tfidf__use_idf': (True, False), 'clf__n_estimators': (20, 25, 50), 'clf__max_features': (100, 500, 1000)}
    find_best_paramas_for_model(RF_classifier, parameters, train_data, train_target)
    


                  

############################## Helper functions ############################## 

def find_best_paramas_for_model(classifier, parameters, X_train, Y_train):
    
    gs_classifier = GridSearchCV(classifier, parameters)
    gs_classifier.fit(X_train, Y_train)
    
    print('Best score: ', gs_classifier.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_classifier.best_params_[param_name]))    

        
def show_topN(classifier, vectorizer, categories, N):
    
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topN = np.argsort(classifier.coef_[i])[-N:]
        print("\n%s: %s" % (category, " ".join(feature_names[topN])))

class Tokenizer(object):
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        tokens = [self.lemmatizer.lemmatize(t) for t in word_tokenize(doc)]
        tokens = [i for i in tokens if i not in string.punctuation]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return tokens    

        
                
if __name__ == '__main__':
	main()