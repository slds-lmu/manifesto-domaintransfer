'''
####################################################
Functions and packages for TFIDF
####################################################
'''



'''
Load packages
'''
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import cpu_count
from sklearn.dummy import DummyClassifier
cpus = cpu_count() - 1





'''
Load data

def read_in_data(text_variable, path_file):
    
    # read in csv
    data = pd.read_csv(path_file, index_col = [0])

    # as string
    data['topic_8'] = data['topic_8'].astype(str) # categories
    data[text_variable] = data[text_variable].astype(str) # topic

    # save appropriate columns as X & Y
    X = data[text_variable] 
    Y = data['topic_8']

    return(data, X, Y)

'''




'''
Evaluate model 
'''
def evaluation_table(Y, Y_pred, domain= 'within-domain', data_description ="training"): 
    #from sklearn.metrics import balanced_accuracy_score
    
    #### scores sklearn
    metrics = {'metric': ['Accuracy', 'F1 score (macro)'],
               'score': [accuracy_score(Y, Y_pred), f1_score(Y, Y_pred, average='macro')]}
    
    metrics = pd.DataFrame(metrics)
    metrics['domain'] = domain
    metrics['data'] = data_description
    return(metrics)
    






'''
inspired by https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/natural_language_processing/example_text_classification.ipynb
More detailed evaluate model per category 
* table [ precision    recall  f1-score   support]
* Confustion matrix
* ROC curve
* Percision recall curve
'''
# template from https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/natural_language_processing/nlp_utils.py
def evaluation_per_category(y_test, predicted, predicted_prob, classes, data_description, path_results, figsize=(30,15)):
    plt.rc('axes', titlesize=40) #fontsize of the title
    plt.rc('axes', labelsize=40) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels

    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    print("Accuracy:",  round(accuracy,2))
    print("Auc:", round(auc,2))
    print("Detail:")
    classification_report = metrics.classification_report(y_test, predicted, output_dict=True)
    classification_report = pd.DataFrame(classification_report).iloc[:,0:8].T
    classification_report['data'] = data_description
    print(classification_report)
    
    ## Plot confusion matrix
    fig = plt.figure() 
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=True, annot_kws={"size": 30})
    ax.set(xlabel="Predicted category", ylabel="True category", xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    #plt.subplots_adjust(top = 5, bottom=4)
    fig.savefig(fname=path_results + '/confusion_matrix_' +data_description + '.png', bbox_inches="tight")
    
    
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', 
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    
    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()
    return(classification_report)