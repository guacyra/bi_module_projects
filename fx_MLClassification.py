#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[ ]:





# In[ ]:


def metrics_train_Class(model, X_train, y_train):
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)
    ypredTrain = model.predict(X_train)
    Acc_train = scores['test_acc'].mean()
    Precision_train = scores['test_prec_macro'].mean()
    Recall_train = scores['test_rec_macro'].mean()
    F1_train = scores['test_f1_macro'].mean()
    conf_matrix_train = confusion_matrix(y_train, ypredTrain)
    from sklearn.metrics import classification_report
    statist_train = []
   
    list_metrics = [Acc_train, Precision_train, Recall_train, F1_train]
    statist_train.append(list_metrics)
    statist_train = pd.DataFrame(statist_train,columns = ['Accuracy', 'Precision', 'Recall', 'f1'], index = ['Train'])
    
    print('-----------------------------------------')
    print('TRAIN results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_train)
    return statist_train


# In[ ]:


def metrics_test_Class(model, X_test, y_test):
    
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_test, y_test, cv=10, scoring=scoring)
    ypredtest = model.predict(X_test)
    report = classification_report(y_test, ypredtest,zero_division=0, output_dict=True)
    report = pd.DataFrame(report).T
    
    Acc_test = report.loc['accuracy', :].mean()  
    Rest_metrics = report.iloc[:-3,:]
    
    Precision_test = Rest_metrics.loc[:,'precision'].mean()
    Recall_test = Rest_metrics.loc[:,'recall'].mean()
    F1_test = Rest_metrics.loc[:,'f1-score'].mean()
    conf_matrix_test = confusion_matrix(y_test, ypredtest)
    
    statist_test = []
   
    list_metrics = [Acc_test, Precision_test, Recall_test, F1_test]
    statist_test.append(list_metrics)
    statist_test = pd.DataFrame(statist_test,columns = ['Accuracy', 'Precision', 'Recall', 'f1'], index = ['test'])
     
    print('-----------------------------------------')
    print('TEST results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_test)
    print(' Classification report \n', Rest_metrics)
    return statist_test


# In[ ]:


def AllmetricsClass(model,X_train,y_train,X_test,y_test):
    
    stats_train = metrics_train_Class(model, X_train,y_train)
    stats_test = metrics_test_Class(model, X_test,y_test)
    final_metrics = pd.concat([stats_train,stats_test])
    print()
    print('++++++++ Summary of the Metrics +++++++++++++++++++++++++++++++++++')
    print(final_metrics)
    return final_metrics


# In[ ]:





# In[ ]:




