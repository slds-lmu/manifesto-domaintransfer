import transformers
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as pltY
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from datasets.dataset_dict import DatasetDict
from datasets import Dataset

'''
Load data

def read_in_data(text_variable, path_file):
    # read in csv
    data = pd.read_csv(path_file,  index_col=None, sep=";")

    # as string
    data['topic_8'] = data['topic_8'].astype(str) # categories
    data[text_variable] = data[text_variable].astype(str) # topic

    return(data)
'''


'''
Evaluate model 
'''
def evaluation_table(Y, Y_pred, domain= 'within-domain', data_description ="training"): 
    
    #### scores sklearn
    metrics = {'metric': ['Accuracy', 'F1 score (macro)'],
               'score': [accuracy_score(Y, Y_pred), f1_score(Y, Y_pred, average='macro')]}
    
    metrics = pd.DataFrame(metrics)
    metrics['domain'] = domain
    metrics['data'] = data_description
    return(metrics)
    



'''
Compute metrics for training
'''
def compute_metrics(eval_pred):
    metric = load_metric('f1')
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return  metric.compute(predictions=predictions, references=labels, average="macro")




'''
Compute predictions
'''
# https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python
#def get_prediction(text, tokenizer):
    # prepare our text into tokenized sequence
    #inputs = tokenizer(text, truncation=True, return_tensors="pt").to("cuda")
    # perform inference to our model
    #outputs = model(**inputs)
    # get output probabilities by doing softmax
    #probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    #return classes[probs.argmax()]




'''
More detailed evaluate model per category 
* table [ precision    recall  f1-score   support]
* Confustion matrix
'''
# template from https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/natural_language_processing/nlp_utils.py
def evaluation_per_category(y_test, predicted, data_description, path_results, figsize=(30,15)):
    plt.rc('axes', titlesize=40) #fontsize of the title
    plt.rc('axes', labelsize=40) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels

    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    classes = pd.get_dummies(y_test, drop_first=False).columns
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:",  round(accuracy,2))
    print("Detail:")
    classification_report = metrics.classification_report(y_test, predicted, output_dict=True)
    classification_report = pd.DataFrame(classification_report).iloc[:,0:8].T
    classification_report['data'] = data_description
    classification_report.to_csv(path_results + '/classification_report_detailed_' +  data_description +'.csv', index=True)
    #print(classification_report)
    
    ## Plot confusion matrix
    fig = plt.figure() 
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=True, annot_kws={"size": 30})
    ax.set(xlabel="Predicted category", ylabel="True category", xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    #plt.subplots_adjust(top = 5, bottom=4)
    fig.savefig(fname= path_results + '/confusion_matrix_' + data_description + '.png', bbox_inches="tight")
    return(classification_report)



'''
predict classes of trained BERT model as categories (not numerical dummy encoding)

def predict_BERT(encoded_dataset, dic_y_mapping, Y_train):
  #https://discuss.huggingface.co/t/different-results-predicting-from-trainer-and-model/12922/4
  #https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
  # Make prediction
  raw_pred, _, _ = trainer.predict(encoded_dataset)

  # Preprocess raw predictions
  Y_pred = np.argmax(raw_pred, axis=1)

  # prediction
  Y_pred = [dic_y_mapping[y_train] for y_train in Y_pred]
  Y_pred = np.array(Y_pred, dtype= 'object')

  # true values
  Y_classes = [dic_y_mapping[y_train] for y_train in Y_train]
  Y_classes = np.array(Y_classes, dtype= 'object')
  return Y_pred, Y_classes
'''


'''
More detailed evaluate model per category 
* table [ precision    recall  f1-score   support]
* Confustion matrix
'''
# template from https://github.com/mdipietro09/DataScience_ArtificialIntelligence_Utils/blob/master/natural_language_processing/nlp_utils.py
def evaluation_per_category(y_test, predicted, data_description, path_results, figsize=(30,15)):
    plt.rc('axes', titlesize=40) #fontsize of the title
    plt.rc('axes', labelsize=40) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=30) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=30) #fontsize of the y tick labels

    classes = ['economy', 'external relations', 'fabric of society', 'freedom and democracy', 
               'no topic', 'political system', 'social groups', 'welfare and quality of life']
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    print("Accuracy:",  round(accuracy,2))
    print("Detail:")
    classification_report = metrics.classification_report(y_test, predicted, output_dict=True)
    classification_report = pd.DataFrame(classification_report).iloc[:,0:8].T
    classification_report['data'] = data_description
    classification_report.to_csv(path_results + '/classification_report_detailed_' +  data_description +'.csv', index=True)
    #print(classification_report)
    
    ## Plot confusion matrix
    fig = plt.figure() 
    cm = metrics.confusion_matrix(y_test, predicted, labels = classes)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=True, annot_kws={"size": 30})
    ax.set(xlabel="Predicted category", ylabel="True category", xticklabels=classes, yticklabels=classes)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    #plt.subplots_adjust(top = 5, bottom=4)
    fig.savefig(fname= path_results + '/confusion_matrix_' + '_' + data_description + '.png', bbox_inches="tight")
    return(classification_report)






'''
evaluate model per country
 
def evaluation_country(countryname, X_country, y, y_pred):
  country_seclected = X_country == countryname 
  acc = accuracy_score(y[country_seclected],  np.array(y_pred)[country_seclected])
  f1 = f1_score(y[country_seclected],  np.array(y_pred)[country_seclected], average='macro')
  return(acc, f1)
'''


'''
evaluate model per country

def evaluation_country(data, countryname, X_country, y, y_pred):
  country_seclected = X_country == countryname 
  acc = accuracy_score(y[country_seclected],  np.array(y_pred)[country_seclected])
  f1 = f1_score(y[country_seclected],  np.array(y_pred)[country_seclected], average='macro')
  no_topics = int(data.query("countryname == '" + countryname + "' & topic_8 =='no topic'").shape[0])
  observations = int(data.query("countryname == '" + countryname + "'").shape[0])
  percentage_no_topics = round(no_topics/observations, 2)
  return(acc, f1, observations, no_topics, percentage_no_topics)
'''


'''
evaluate model per country
'''
def evaluation_country(countryname, X_country, y, y_pred):
  country_seclected = X_country == countryname 
  acc = accuracy_score(y[country_seclected],  np.array(y_pred)[country_seclected])
  f1 = f1_score(y[country_seclected],  np.array(y_pred)[country_seclected], average='macro')
  no_topics = sum((X_country == countryname) & (y == 'no topic'))
  observations = sum((X_country == countryname))
  percentage_no_topics = round(no_topics/observations, 2)
  return(acc, f1, observations, no_topics, percentage_no_topics)



'''
evaluate model per language

def evaluation_language(language, X_language, y, y_pred):
  language_seclected = X_language== language 
  acc = accuracy_score(y[language_seclected],  np.array(y_pred)[language_seclected])
  f1 = f1_score(y[language_seclected],  np.array(y_pred)[language_seclected], average='macro')
  no_topics = int(data.query("language == '" + language + "' & topic_8 =='no topic'").shape[0])
  observations = int(data.query("language == '" + language + "'").shape[0])
  percentage_no_topics = round(no_topics/observations, 2)
  return(acc, f1, observations, no_topics, percentage_no_topics)
'''

'''
evaluate model per language
'''
def evaluation_language(language, X_language, y, y_pred):
  language_seclected = X_language== language 
  acc = accuracy_score(y[language_seclected],  np.array(y_pred)[language_seclected])
  f1 = f1_score(y[language_seclected],  np.array(y_pred)[language_seclected], average='macro')
  no_topics = sum((X_language == language) & (y == 'no topic'))
  observations = sum((X_language == language))
  percentage_no_topics = round(no_topics/observations, 2)
  return(acc, f1, observations, no_topics, percentage_no_topics)

'''
Encode data set
'''
def preprocess_function(examples):
  return tokenizer(examples['text'], truncation=True)
# Note truncation=True: input truncated to the maximum length accepted by the model


'''
get predictions from BERD model
'''
def predict_BERT(encoded_dataset, dic_y_mapping, Y_train):
  #https://discuss.huggingface.co/t/different-results-predicting-from-trainer-and-model/12922/4
  #https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
  # Make prediction
  raw_pred, _, _ = trainer.predict(encoded_dataset)

  # Preprocess raw predictions
  Y_pred = np.argmax(raw_pred, axis=1)

  # prediction
  Y_pred = [dic_y_mapping[y_train] for y_train in Y_pred]
  Y_pred = np.array(Y_pred, dtype= 'object')

  # true values
  Y_classes = [dic_y_mapping[y_train] for y_train in Y_train]
  Y_classes = np.array(Y_classes, dtype= 'object')
  return Y_pred, Y_classes
