#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:17:39 2019

@author: Ashish Chouhan
Description:
    Stop words removal service for Natural Language Processing
"""
from import_library import *

def remove_stop_words(use_case_words):
    # Remove Stop Words
    stop_words = set(stopwords.words("english"))
    filtered_sentence = [w for w in use_case_words if not w in stop_words and len(w) > 3]
    return filtered_sentence
