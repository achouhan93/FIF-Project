# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:21:46 2019

@author: Ashish Chouhan
Description:
    Feature selection filter method with correlation
"""
from import_library import *
from Counter_Processing import processing_array_generated

def feature_correlation(X_features, Y_target, no_of_features):
    
    # ---------------------------------------------------------------------------------- #        
    correlated_features = []
    
    X_features.reset_index(drop=True, inplace=True)
    Y_target.reset_index(drop=True, inplace=True)
    complete_data = pd.concat([X_features, Y_target], axis = 1)    
    
    existing_correlation_technique = ['pearson', 'kendall' , 'spearman']
    
    for correlation_technique in existing_correlation_technique:
        correlation_matrix = complete_data.corr(method=correlation_technique)
        position = correlation_matrix.columns.get_loc(Y_target.columns[0])
    
        # Extract Correlated features
        for i in range(len(correlation_matrix .columns)):  
            if abs(correlation_matrix.iloc[position, i]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.append(colname)
    
    correlated_features_finalised = processing_array_generated(correlated_features, len(X_features.columns))      
    correlated_features_dataframe = complete_data.loc[:, correlated_features_finalised]
    correlated_features_dataframe = pd.DataFrame(correlated_features_dataframe)
    return (correlated_features_finalised, correlated_features_dataframe)
