# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:48:19 2019

@author: ashis
"""
from import_library import *

def duplicate_removal(table_data):
    table_data.fillna(0, inplace=True)
    table_data_T = table_data.T
    table_data = table_data_T.drop_duplicates(keep='first').T
    return table_data

