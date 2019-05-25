# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:53:40 2019

@author: Ashish Chouhan
"""
from import_library import *

def word_embedding_tfidf(reference_array,comparison_array):
    updated_reference_array = []
        
    for field in reference_array:
        field = re.sub('[^0-9a-zA-Z]+', ' ', field)
        updated_reference_array.append(field)
    
    updated_reference_array = pd.unique(updated_reference_array).tolist()
        
    array_list = np.asarray(comparison_array)
    comparison_string = np.array_str(array_list)
        
    comparison_string = re.sub('_', ' ', comparison_string)
    updated_reference_array.append(comparison_string)
        
    TfidVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidVec.fit_transform(updated_reference_array)
    
    updated_reference_array.remove(comparison_string)
        
    return tfidf
    
