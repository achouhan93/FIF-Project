#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:10:08 2019

@author: Ashish Chouhan
"""
from import_library import *

def scaling(column_data):
    sc = StandardScaler()
    df_scaled = sc_x.fit_transform(column_data)
    df_scaled = pd.DataFrame(df_scaled, index = column_data.index, columns = column_data.columns)
    return df_scaled




