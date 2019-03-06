# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:44:39 2018

@author: Ashish Chouhan
"""
#nltk.download('all')
from nltk.stem import PorterStemmer

# Import Libraries required for Processing
import pandas as pd
import numpy as np

# Tokenization
import nltk

from nltk.tokenize import sent_tokenize , word_tokenize

# Stop Words    
from nltk.corpus import stopwords

from nltk import ne_chunk, pos_tag

# Setting file names used during execution process
# Input File :  Use Cases as a text file
#               Walmart_Metadata file for metadata information
file_input_use_case = 'Use Cases to Analyse.txt'
file_input_metadata = 'Walmart_Database_Metadata.xlsx'

# Read Input File to fetch  Information of Use Case and Database Metadata information 

df_input_use_case = pd.read_table(file_input_use_case, header = 0, dtype={'Use Case':np.str})
df_input_metadata = pd.read_excel(file_input_metadata, header = 0)


# Select only relevant columns from Metadata Information
df_metadata = df_input_metadata.loc[ : , ['Field', 'Comment'] ]

# NLP processing for use case

# Tokenizer
example_text = df_input_use_case.applymap(str).iloc[0:1]
example_text1 = df_input_use_case.get_value(0, 'Use Case')

# print(example_text1)
# print(sent_tokenize(example_text1))
# print(word_tokenize(example_text1))

# for i in word_tokenize(example_text1):
#    print(i)

stop_words = set(stopwords.words("english"))
 
# print(stop_words)

words= word_tokenize(example_text1)
# 
# filtered_sentence = []
# 
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
#         

filtered_sentence = [w for w in words if not w in stop_words]

# sent_tokenize is used to create token of a sentence
# sents = sent_tokenize(text)
# print(sents)

# word_toeknize is used to create token of a word in the sentence
# words = word_tokenize(text)
# print(words)

# this will token . also in the statement
# print(nltk.wordpunct_tokenize(example_text1))

# -------------------************ ---------------
# Extract the part of speech of the word in the sentence

# print(nltk.pos_tag(words))
print(example_text1)
print(filtered_sentence)
print(nltk.pos_tag(filtered_sentence))

ps = PorterStemmer()

for w in filtered_sentence:
    print(ps.stem(w))

# Extract Enitiy present in the statement -
    # Tokenization
    # Stop Words
    # Stemming *
    # Part of Speech Tagging
    # Name Entity Recognition

def entities(text):
    return ne_chunk(
            pos_tag(
                    word_tokenize(text)), binary = True)

tree = entities(example_text1)
print(tree)
tree.pprint()
# tree.draw()

