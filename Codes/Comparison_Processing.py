# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:53:40 2019

@author: Ashish Chouhan
"""
from import_library import *

class comparison_values():
    
    def similar_values(reference_array,comparison_array, number_of_values, comparison_technique):
        updated_reference_array = []
        
        for field in reference_array:
            field = re.sub('[^0-9a-zA-Z]+', ' ', field)
            updated_reference_array.append(field)
    
        updated_reference_array = pd.unique(updated_reference_array).tolist()
        
        array_list = np.asarray(comparison_array)
        comparison_string = np.array_str(array_list)
        
        comparison_string = re.sub('_', ' ', comparison_string)
        #reference_array.append(comparison_string)
        updated_reference_array.append(comparison_string)
        
        TfidVec = TfidfVectorizer(stop_words='english')
        #tfidf = TfidVec.fit_transform(reference_array)
        tfidf = TfidVec.fit_transform(updated_reference_array)
        
        distance_value = pairwise_distances(tfidf[-1], tfidf, metric = comparison_technique)
        
        matching_values = []
        
        for i in range(1, number_of_values + 1):
            index = distance_value.argsort()[0][i]
            matching_values.append(reference_array[index])
        
        #reference_array.remove(comparison_string)
        updated_reference_array.remove(comparison_string)
        
        if number_of_values == 1:
            return matching_values[0]
        else:
            return matching_values
    
    def processing_array_generated(combined_list, number_of_values):
        most_count_values = Counter(combined_list)
        
        most_relevant_values =  most_count_values.most_common(number_of_values)
        
        if number_of_values == 1:
            return most_relevant_values[0][0]
        else:
            column = []
            for i in range(len(most_relevant_values)):
                column.append(most_relevant_values[i][0])
            
            return column
