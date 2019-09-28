#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:10:12 2019

@author: ashish
"""
from import_library import *

def logistic_regression_classification(X_train, X_test, Y_train, Y_test):
    
    model = LogisticRegression(random_state=0, solver='lbfgs',
                               multi_class='multinomial', n_jobs=-1)
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = LogisticRegression(random_state=0, solver='lbfgs',
                                       multi_class='multinomial', n_jobs=-1)
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = accuracy_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())