#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:47:48 2019

@author: Ashish Chouhan
Label Encoding with Label Encoder
"""
from import_library import *

def label_encoding(column_data):
    labelencoder = LabelEncoder()
    column_data = labelencoder.fit_transform(column_data.astype(str))
    return column_data
    
    

