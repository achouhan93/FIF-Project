#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:54:42 2019

@author: Ashish Chouhan

"""
from import_library import *

def spell_checker(tokenized_words):
    
    spell_checked_words = []
    # Tokenizer
    for word in tokenized_words:
        w = Word(word)
        correct_word = w.spellcheck()
        
        spell_checked_words.append(correct_word[0][0])
        
    return spell_checked_words