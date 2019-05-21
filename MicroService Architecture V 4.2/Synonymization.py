#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:19:55 2019

@author: Ashish Chouhan
Description: 
    Synonymization Process involved in Natural Language Processing
"""
from import_library import *

def synonyms_words(important_words):
    synonyms = []
        
    for i in range(len(important_words)):
        for syn in wordnet.synsets(important_words[i]):
            for l in syn.lemmas():
                if len(l.name()) > 3:
                    synonyms.append(l.name())
        
    return np.unique(synonyms)

