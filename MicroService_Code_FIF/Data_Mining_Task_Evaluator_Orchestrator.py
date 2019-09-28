#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:37:27 2019

@author: Ashish Chouhan
Description:
    Data Mining Task Evaluator Orchestrator
"""
from import_library import *
from Linear_Regression import linear_regression
from Support_Vector_Machine import SVM_regression, SVM_classification
from Decision_Tree import decision_tree_regression, decision_tree_classification
from Random_Forest import random_forest_regression, random_forest_classification
from K_Nearest_Neighbors import knn_classification
from Naive_Bayes import naive_bayes_classification
from Stochastic_Gradient import SGD_classification
from Logistic_Regression import logistic_regression_classification

def algorithm_selection_processing(relevant_columns, lda_output, feature_encoded):
    
    labeled_feature = [] 
    
    if lda_output[0] == "Classification":
        for value in range(len(feature_encoded)):
            labeled_feature.append(feature_encoded[value][0] + '___' + feature_encoded[value][1] + '___' + feature_encoded[value][2])
        
        labeled_feature = list(set(labeled_feature))
        for j in range(len(labeled_feature)):
            if labeled_feature[j] in relevant_columns.columns:
                relevant_columns = relevant_columns.astype({labeled_feature[j]:'int64'}, errors='ignore')
                
    algorithm_details = []
               
    for i in range(len(relevant_columns.columns)):
        column_data = relevant_columns.copy()
        train_features, test_features, train_labels, test_labels = train_test_split(  
                    column_data.drop(labels=[column_data.columns[i]], axis=1),
                    column_data[column_data.columns[i]],
                    test_size=0.2,
                    random_state=41)
        
        if lda_output[0] == "Regression":
            if lda_output[1] == "LinearRegression":
                accuracy_percent, target_columns, features_columns  = linear_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Linear Regression", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == "SVMRegression":
                accuracy_percent, target_columns, features_columns  = SVM_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Support Vector Regressor", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == "DecisionTree":
                accuracy_percent, target_columns, features_columns  = decision_tree_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Decision Tree Regressor", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == "RandomForest":
                accuracy_percent, target_columns, features_columns  = random_forest_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Random Forest Regressor", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)               
            elif lda_output[1] == " ":
                accuracy_percent, target_columns, features_columns  = linear_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Linear Regression", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = SVM_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Support Vector Regressor)", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                                                
                accuracy_percent, target_columns, features_columns  = decision_tree_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Decision Tree Regressor", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = random_forest_regression(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Random Forest Regressor", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
        elif lda_output[0] == "Classification":
            if lda_output[1] == " " and test_labels.dtypes != 'float64':

                accuracy_percent, target_columns, features_columns  = knn_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Logistic Regression", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = decision_tree_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Decision Tree Classifier", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)

                accuracy_percent, target_columns, features_columns  = SGD_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Stochstic Gradient Descent Classifier", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = SVM_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Support Vector Machine", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)

                accuracy_percent, target_columns, features_columns  = random_forest_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Random Forest", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)                
                
                accuracy_percent, target_columns, features_columns  = naive_bayes_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Naive Bayes", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = logistic_regression_classification(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Logistic Regression", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
               
    algorithms = []
    accuracy_score = []
    target_values = []
    independent_values = []        

    algorithm_details = sorted(algorithm_details, key = lambda x: x[1], reverse = True)     

    if algorithm_details:
    
        for i in range(len(algorithm_details)):
            algorithms.append(algorithm_details[i][0])
            accuracy_score.append(algorithm_details[i][1])
            target_values.append(algorithm_details[i][2])
            independent_values.append(algorithm_details[i][3])
        
        message = " "
    else:
        message = "Hypothesis has no feature which can be considered as Dependent feature for Model."
        
    return (algorithms, accuracy_score, target_values, independent_values, message)


