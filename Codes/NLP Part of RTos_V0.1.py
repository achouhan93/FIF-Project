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
# example_text = df_input_use_case.applymap(str).iloc[0:1]
example_text1 = df_input_use_case.get_value(0, 'Use Case')
words= word_tokenize(example_text1)         

stop_words = set(stopwords.words("english"))
filtered_sentence = [w for w in words if not w in stop_words]
print(words)
print(nltk.pos_tag(words))

pos_tag_filtered_sentence = nltk.pos_tag(words)
#pos_tag_filtered_sentence = nltk.pos_tag(filtered_sentence)
namedEnt = ne_chunk(pos_tag_filtered_sentence)

chunkGram = r"""Chunk: {<.*>+}
                                        }<VB.?|IN|DT|CC|CD|,>+{"""

chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(namedEnt)

print(chunked)

for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
    print(subtree)



