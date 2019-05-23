#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:46:40 2019

@author: Ashish Chouhan
Description:
    Counter Processing
"""
from import_library import *

def processing_array_generated(combined_list, number_of_values):
    most_count_values = Counter(combined_list)
        
    most_relevant_values =  most_count_values.most_common(number_of_values)
    column = []
    if number_of_values == 1:
        column.append(most_relevant_values[0][0])
    else:
        for i in range(len(most_relevant_values)):
            column.append(most_relevant_values[i][0])

    return column
