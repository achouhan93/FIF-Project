# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:57:10 2019

@author: ashis
"""
from import_library import *
from Tokenizer import tokenize
from Spelling_Corrector import spell_checker
from Stop_Words_Removal import remove_stop_words
from Synonymization import synonyms_words
from Part_of_Speech_Tagging import part_of_speech_tagging

def clean_text(text):
    # NLP Pre-Processing 
    tokenize_words = tokenize(text.lower())
    stop_words_removed = remove_stop_words(tokenize_words)
    relevant_words = [WordNetLemmatizer().lemmatize(words, pos='v') for words in stop_words_removed if len(words) > 3]
    return relevant_words

def lda_supervised_topic_modelling(stop_words_removed):
    
    lda_output = []
    relevant_words = [WordNetLemmatizer().lemmatize(words, pos='v') for words in stop_words_removed if len(words) > 3]
    dict_FIF = corpora.Dictionary.load('dictionary_FIF')
    lda_model =  models.LdaModel.load('lda_model')

    myTopicsList = []

    for idx, topic in lda_model.print_topics(5):
        myTopicsList.append(re.findall(r"\"(.*?)\"",lda_model.print_topic(idx, 5)))

    searchClassification = " classification discrete labels class category observation branch group separate decision sets divided segmentation segregate"
    searchRegression = "regression linear analyze distribution predict statisticsapproximate continuous range quantity series forecasting estimate count"

    regressionCount = 0
    classificationCount = 0

    bow = dict_FIF.doc2bow(relevant_words)
    ldaData = lda_model[bow]

    if ldaData[0][1] == ldaData[1][1]:
        lda_output.append(" ")
    else:
        sort = sorted(ldaData, key=lambda tup:(-tup[1], tup[0]))
        topic_num = (sort[0][0])
           
        topicKeywords =myTopicsList[topic_num]
        
        for word in topicKeywords:
            if word in searchRegression:
                regressionCount+=1
            if word in searchClassification:
                classificationCount+=1
       
        if regressionCount > classificationCount:
            lda_output.append("Regression")
        else:
            lda_output.append("Classification")
    
    if lda_output[0] == "Regression":
        subPart = lda_regression_topic_modelling(relevant_words)
        lda_output.append(subPart)
    else:
        lda_output.append(" ")
        
    return lda_output

def lda_regression_topic_modelling(relevant_words):
    subPart = " "

    regression_dict = corpora.Dictionary.load('regression_dictionary_FIF')
    lda_regression_model =  models.LdaModel.load('regression_lda_model')

    myTopicsList = []

    for idx, topic in lda_regression_model.print_topics(5):
        myTopicsList.append(re.findall(r"\"(.*?)\"",lda_regression_model.print_topic(idx, 5)))

    searchSVMregression = "support vector SVR SVM maximal margin nonlinear kernel feature space dimensionality soft epsilon multipliers"
    searchlinearRegression = "mean root square correlation slope line error linear equation gradient descent average scatterplot bias"

    linearregressionCount = 0
    svmregressionCount = 0

    bow = regression_dict.doc2bow(relevant_words)
    ldaData = lda_regression_model[bow]

    if ldaData[0][1] != ldaData[1][1]:

        sort = sorted(ldaData, key=lambda tup:(-tup[1], tup[0]))
        topic_num = (sort[0][0])
        topics = lda_regression_model.print_topic(topic_num, 5)
    
        topicKeywords =myTopicsList[topic_num]

        for word in topicKeywords:
            if word in searchlinearRegression:
                linearregressionCount+=1
            if word in searchSVMregression:
                svmregressionCount+=1
       
        if linearregressionCount > svmregressionCount:
            subPart = "LinearRegression"
        else:
            subPart = "SVMRegression"
        
    return subPart
