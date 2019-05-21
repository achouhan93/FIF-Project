# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:21:46 2019

@author: ashis
"""
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
    
    return correlated_features
