# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:09:48 2019

@author: ashis
"""
from import_library import *

def variance_feature(table_data, logs):
    # Constant Columns Removal
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(table_data)       
    
    # Remove the constant columns
    constant_columns = [column for column in table_data.columns  
                    if column not in table_data.columns[constant_filter.get_support()]]
    table_data.drop(labels=constant_columns, axis=1, inplace=True)
    
    if constant_columns:
        log = (",".join(constant_columns)) + " : Constant Columns are removed"
        logs.append(log)
        
    # Quasi-Constant Columns Removal
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(table_data)
    # Remove the Quasi-Constant Columns
    qconstant_columns = [column for column in table_data.columns  
                    if column not in table_data.columns[qconstant_filter.get_support()]]
    table_data.drop(labels=qconstant_columns, axis=1, inplace=True)
    
    if qconstant_columns:
        log = (",".join(qconstant_columns)) + " : Quasi-Constant Columns are removed"
        logs.append(log)
    
    return (table_data, logs)