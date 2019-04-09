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

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector

import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
