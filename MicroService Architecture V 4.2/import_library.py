# -*- coding: utf-8 -*-
"""
Created on Thursday 14th February 2019
@author: Ashish Chouhan
Description:
    RToS Implementation without LDA but all Algorithm Covered
    (Micro Service Architecture)
"""
# Import Libraries required for Processing
import sys
import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter

# NLP Processing Library
# Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
# Stop Words    
from nltk.corpus import stopwords
# Word Net
from nltk.corpus import wordnet
#Spelling Corrector
from textblob import Word

# Evaluation Library Import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# Database Related Library
import MySQLdb

# Table Display Library
from prettytable import PrettyTable

# Filter Methods Library
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

#LDA Implementation
from gensim import models, corpora
from nltk.stem import WordNetLemmatizer

from mlxtend.feature_selection import ExhaustiveFeatureSelector