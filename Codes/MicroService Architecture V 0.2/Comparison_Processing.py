# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:53:40 2019

@author: Ashish Chouhan
"""
from import_library import *

class comparison_values():
    
    def processing_array_generation(reference_array,comparison_array):
        array_list = np.asarray(comparison_array)
        comparison_string = np.array_str(array_list)
        reference_array.append(comparison_string)
        TfidVec = TfidfVectorizer(stop_words='english')
        tfidf = TfidVec.fit_transform(reference_array)
        
        
        vals = cosine_similarity(tfidf[-1], tfidf, dense_output=True)
        idx = vals.argsort()[0][-2]
        
        vals1 = euclidean_distances(tfidf[-1], tfidf)
        idx1 = vals1.argsort()[0][1]
        
        reference_array.remove(comparison_string)
        
        return reference_array[idx1]
    
    


