#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.metrics import r2_score


# In[ ]:


def metrics_train(X_train,y_train, model):
    
    # Let's calculate the metrics with our TRAIN dataset
    y_predTrain = model.predict(X_train) 
    cv_scores = cross_val_score(model, X_train, 
                            y_train,cv=10, scoring='r2') # Let's define the K and the 

    cv_scores= round(np.mean(cv_scores),3)
    
    statist_train = []
    MAPE_lTrain = metrics.mean_absolute_percentage_error(y_train, y_predTrain)
    MAE_lTrain = metrics.mean_absolute_error(y_train, y_predTrain)
    MSE_lTrain = metrics.mean_squared_error(y_train,y_predTrain)
    RMSE_lTrain = np.sqrt(metrics.mean_squared_error(y_train, y_predTrain))
    R2_lTrain = model.score(X_train, y_train)
    
    list_metrics = [MAPE_lTrain, MAE_lTrain, MSE_lTrain, RMSE_lTrain, R2_lTrain,cv_scores]
    statist_train.append(list_metrics)
    statist_train = pd.DataFrame(statist_train,columns = ['MAPE','MAE', 'MSE', 'RMSE', 'R2', 'CV_R2'], index = ['Train'])
    
    return statist_train


# In[ ]:


def metrics_test(X_test,y_test, model):
    # Let's calculate the metrics with our TRAIN dataset
    y_pred = model.predict(X_test)
    
    statist_test = []
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = model.score(X_test,y_test)
    
    list_metrics = [MAPE, MAE, MSE, RMSE, R2]
    statist_test.append(list_metrics)
    statist_test = pd.DataFrame(statist_test,columns = ['MAPE', 'MAE', 'MSE', 'RMSE', 'R2'], index = ['Test'])
    
    return statist_test


# In[ ]:


def Allmetrics(model,X_train,y_train,X_test,y_test):
    
    stats_train = metrics_train(X_train,y_train,model)
    stats_test = metrics_test(X_test,y_test,model)
    final_metrics = pd.concat([stats_train,stats_test])
    return final_metrics

