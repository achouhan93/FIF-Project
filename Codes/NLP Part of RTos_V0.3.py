# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 23:23:10 2018

@author: ashis
"""

# Import Libraries required for Processing
import pandas as pd
import numpy as np

# Tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Stop Words    
from nltk.corpus import stopwords
# Named Entity Relationship
from nltk import ne_chunk
from nltk.stem import PorterStemmer

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

ps = PorterStemmer()
# Tokenizer
# example_text = df_input_use_case.applymap(str).iloc[0:1]
example_text1 = df_input_use_case.get_value(1, 'Use Case')
#words = [ps.stem(word) for sent in sent_tokenize(example_text1) for word in word_tokenize(sent)]         
words = [word for sent in sent_tokenize(example_text1) for word in word_tokenize(sent)]         

## Remove Punctuation
#words_punctuation = [word.lower() for word in words if word.isalpha()]
words_punctuation = [word for word in words if word.isalpha()]

# Remove Stop Words
stop_words = set(stopwords.words("english"))
filtered_sentence = [w for w in words_punctuation if not w in stop_words]
#
#print(words)
#print(nltk.pos_tag(words))

# Part of Speech Tagging
pos_tag_filtered_sentence = nltk.pos_tag(filtered_sentence)

# Named Entity Relationship
namedEnt = ne_chunk(pos_tag_filtered_sentence)

# Chunking to consider only Nouns
chunkGram = r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|CC|CD>+{"""

chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(namedEnt)

regex_example = r"""NP: {<JJ>*<NN>}      
                                        }<VB.?|IN|DT|CC|CD>+{"""
regex_parser = nltk.RegexpParser(regex_example)
word_regex = regex_parser.parse(namedEnt)
#print("***********************")

#print(chunked)

for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
    print(subtree)

print("*********************")

#print(word_regex)

for subtree in word_regex.subtrees(filter=lambda t: t.label() == 'NP'):
    print(subtree)


