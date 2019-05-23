#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:12:16 2019

@author: Ashish Chouhan
Description:
    Tokenizer service for Natural Language Processing
"""

from import_library import *

def tokenize(use_case):
    # Tokenizer
    words = [word for sent in sent_tokenize(use_case) for word in word_tokenize(sent)]         
    
    # Remove Punctuation
    words_without_punctuation = [word for word in words if word.isalpha()]
    return words_without_punctuation