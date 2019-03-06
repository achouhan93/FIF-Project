# -*- coding: utf-8 -*-
"""
Created on Thursday 14th February 2019
@author: Ashish Chouhan
Description:
    RToS Implementation without LDA but all Algorithm Covered
    (Micro Service Architecture)
"""
from import_library import *

class nlp_pre_process():
    
    def tokenize(use_case):
        # Tokenizer
        words = [word for sent in sent_tokenize(use_case) for word in word_tokenize(sent)]         
        return words
    
    def remove_punctuation(tokenize_use_case):
        # Remove Punctuation
        words_without_punctuation = [word for word in tokenize_use_case if word.isalpha()]
        return words_without_punctuation
    
    def remove_stop_words(use_case_words):
        # Remove Stop Words
        stop_words = set(stopwords.words("english"))
        filtered_sentence = [w for w in use_case_words if not w in stop_words]
        return filtered_sentence
    
    def part_of_speech_tagging(filtered_use_case_words):
        # Part of Speech Tagging
        pos_tag_words = nltk.pos_tag(filtered_use_case_words)
        return pos_tag_words

