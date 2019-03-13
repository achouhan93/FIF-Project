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
    
    def important_words_extraction(pos_tag_words):
        regex_example = r"""NP: {<NN.*>?<NN.*>}      
                                        }<VB.?|IN|DT|CC|CD>+{"""
        regex_parser = nltk.RegexpParser(regex_example)
        word_regex = regex_parser.parse(pos_tag_words)
        d = []
        
        for subtree in word_regex.subtrees(filter=lambda t: t.label() == 'NP'):
            d.append(subtree)
        
        final_word =[]
        for i in range(len(d)):
            for j in range(len(d[i])):
                final_word.append(d[i][j][0])
        
        return np.unique(final_word)
    
    def synonyms_words(important_words):
        synonyms = []
        
        for i in range(len(important_words)):
            for syn in wordnet.synsets(important_words[i]):
                for l in syn.lemmas():
                    synonyms.append(l.name())
        
        return np.unique(synonyms)

