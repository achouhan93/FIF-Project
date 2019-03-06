# -*- coding: utf-8 -*-
"""
Created on Thursday 14th February 2019
@author: Ashish Chouhan
Description:
    RToS Implementation without LDA but all Distances Covered
"""

# Import Libraries required for Processing
import pandas as pd
import numpy as np

# Tokenization
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Stop Words    
from nltk.corpus import stopwords

# Word Net
from nltk.corpus import wordnet

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
df_comment = df_input_metadata['Comment'].tolist()
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
# ***************************************************************************** #

# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(filtered_sentence)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(df_comment.split())]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))

# ***************************************************************************** #
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

synonyms = []

for i in range(len(final_word)):
    for syn in wordnet.synsets(final_word[i]):
        for l in syn.lemmas():
            synonyms.append(l.name())
            
print(len(synonyms)) 
print(len(set(synonyms)))

synonyms_value = np.unique(synonyms)

# ***************************************************************************** #

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

new_array = np.array_str(synonyms_value)

df_comment.append(new_array)

#    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
TfidVec = TfidfVectorizer(stop_words='english')
tfidf = TfidVec.fit_transform(df_comment)
tfidf.shape

print("********************* Cosine Similarity ********************************")

vals = cosine_similarity(tfidf[-1], tfidf)
idx = vals.argsort()[0][-2]
idx1 = vals.argsort()[0][-3]
idx2 = vals.argsort()[0][-4]
idx3 = vals.argsort()[0][-5]
idx4 = vals.argsort()[0][-6]

#flat = vals.flatten()
#flat.sort()
#req_tfidf = flat[-2]

print(example_text1)
print(df_comment[idx])
print(df_comment[idx1])
print(df_comment[idx2])
print(df_comment[idx3])
print(df_comment[idx4])
# ***************************************************************************** #

print("********************* Euclidean Distances ****************************")

from sklearn.metrics.pairwise import euclidean_distances
vals1 = euclidean_distances(tfidf[-1], tfidf)
idx1 = vals1.argsort()[0][1]
idx11 = vals1.argsort()[0][2]
idx12 = vals1.argsort()[0][3]
idx13 = vals1.argsort()[0][4]
idx14 = vals1.argsort()[0][5]

print(example_text1)
print(df_comment[idx1])
print(df_comment[idx11])
print(df_comment[idx12])
print(df_comment[idx13])
print(df_comment[idx14])

# ***************************************************************************** #

print("******************** Manhattan Distances ******************************")

from sklearn.metrics.pairwise import manhattan_distances
vals2 = manhattan_distances(tfidf[-1], tfidf)
idx2 = vals2.argsort()[0][1]
idx21 = vals2.argsort()[0][2]
idx22 = vals2.argsort()[0][3]
idx23 = vals2.argsort()[0][4]
idx24 = vals2.argsort()[0][5]

print(example_text1)
print(df_comment[idx2])
print(df_comment[idx21])
print(df_comment[idx22])
print(df_comment[idx23])
print(df_comment[idx24])
