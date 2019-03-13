# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:25:52 2019

@author: Ashish
"""
from import_library import *

class feature_selection_script():
    
    def feature_selection_processing(database_fields, database_connection):
        array_list = np.asarray(comparison_array)
        comparison_string = np.array_str(array_list)
        reference_array.append(comparison_string)
        TfidVec = TfidfVectorizer(stop_words='english')
        tfidf = TfidVec.fit_transform(reference_array)
        
        distance_value = pairwise_distances(tfidf[-1], tfidf, metric = comparison_technique)
        
        matching_values = []
        
        for i in range(1, number_of_values + 1):
            index = distance_value.argsort()[0][i]
            matching_values.append(reference_array[index])
        
        reference_array.remove(comparison_string)
        
        if number_of_values == 1:
            return matching_values[0]
        else:
            return matching_values
