#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:08:09 2019

@author: ashish
"""
from import_library import *

def knn_classification(X_train, X_test, Y_train, Y_test):
    
    model = KNeighborsClassifier()
    rfe = ExhaustiveFeatureSelector(model, 
           min_features=1,
           max_features=len(X_train.columns),
           scoring='accuracy',
           print_progress=True,
           cv=5)
    features = rfe.fit(X_train.values, Y_train.values, custom_feature_names = X_train.columns)
    
    filtered_features = []
    for i in range(len(features.best_feature_names_)):
        filtered_features.append(features.best_feature_names_[i])
    
    testing_model = KNeighborsClassifier()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = accuracy_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features) 