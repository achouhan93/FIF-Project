# -*- coding: utf-8 -*-
"""
Created on Thursday 14th February 2019
@author: Ashish Chouhan
Description:
    Part of Speech tagging and Chunking process of Natural Language Processing
"""
from import_library import *

def part_of_speech_tagging(filtered_use_case_words):
    # Part of Speech Tagging
    pos_tag_words = nltk.pos_tag(filtered_use_case_words)
    regex_example = r"""NP: {<NN.*>?<NN.*>|<JJ.*>}      
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
     
    


