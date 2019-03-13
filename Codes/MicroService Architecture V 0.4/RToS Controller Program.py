# -*- coding: utf-8 -*-
"""
Created on Thursday 14th February 2019
@author: Ashish Chouhan
Description:
    RToS Implementation without LDA but all Algorithm Covered
    (Micro Service Architecture)
"""

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
