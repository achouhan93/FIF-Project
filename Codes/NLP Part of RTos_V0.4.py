# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 23:23:10 2018

@author: ashis

POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
"""

# Import Libraries required for Processing
import pandas as pd
import numpy as np

# Tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Stop Words    
from nltk.corpus import stopwords

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

# ***************************************************************************** #
# NLP processing for use case

# Tokenizer
example_text1 = df_input_use_case.get_value(0, 'Use Case')

words = [word for sent in sent_tokenize(example_text1) for word in word_tokenize(sent)]         

# Remove Punctuation
words_punctuation = [word for word in words if word.isalpha()]

# Remove Stop Words
stop_words = set(stopwords.words("english"))
filtered_sentence = [w for w in words_punctuation if not w in stop_words]

# Part of Speech Tagging
pos_tag_filtered_sentence = nltk.pos_tag(filtered_sentence)

regex_example = r"""NP: {<NN.*>?<NN.*>}      
                                        }<VB.?|IN|DT|CC|CD>+{"""
regex_parser = nltk.RegexpParser(regex_example)
word_regex = regex_parser.parse(pos_tag_filtered_sentence)

d = []
for subtree in word_regex.subtrees(filter=lambda t: t.label() == 'NP'):
    d.append(subtree)

final_word =[]

for i in range(len(d)):
    for j in range(len(d[i])):
        final_word.append(d[i][j][0])

Company = final_word[0]
print(Company)

final_word.remove(Company)
print(np.unique(final_word))

# ***************************************************************************** #

