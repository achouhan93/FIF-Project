# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:57:10 2019

@author: ashis
"""
from import_library import *
from NLP_PreProcessing import nlp_pre_process

def clean_text(text):
    # NLP Pre-Processing 
    tokenize_words = nlp_pre_process.tokenize(text.lower())
    punctuation_removed = nlp_pre_process.remove_punctuation(tokenize_words)
    stop_words_removed = nlp_pre_process.remove_stop_words(punctuation_removed)
    relevant_words = [WordNetLemmatizer().lemmatize(words, pos='v') for words in stop_words_removed if len(words) > 3]
    return relevant_words

def lda_supervised_topic_modelling(stop_words_removed):
    
    lda_output = []
    relevant_words = [WordNetLemmatizer().lemmatize(words, pos='v') for words in stop_words_removed if len(words) > 3]
    data = pd.read_excel('FIF_supervised Dataset.xlsx')
    data_text = data[['Details']]
    documents = data_text

    # For gensim we need to tokenize the data and filter out stopwords
    tokenized_data = []
    for x in range(len(documents)):
        text = documents['Details'][x]
        tokenized_data.append(clean_text(text))

    # Build a Dictionary - association word to numeric id
    dictionary = corpora.Dictionary(tokenized_data)

    # later on, load trained model from file
    lda_model =  models.LdaModel.load('lda_model')
    myTopicsList = []

    for idx, topic in lda_model.print_topics(5):
        myTopicsList.append(re.findall(r"\"(.*?)\"",lda_model.print_topic(idx, 5)))

    searchClassification = " classification discrete labels class category observation branch group separate decision sets divided segmentation segregate"
    searchRegression = "regression linear analyze distribution predict statisticsapproximate continuous range quantity series forecasting estimate count"

    regressionCount = 0
    classificationCount = 0

    bow = dictionary.doc2bow(relevant_words)
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
    data = pd.read_excel('FIF_regression Dataset.xlsx')
    data_text = data[['Details']]
    documents = data_text

    # For gensim we need to tokenize the data and filter out stopwords
    tokenized_data = []
    for x in range(len(documents)):
        text = documents['Details'][x]
        tokenized_data.append(clean_text(text))

    # Build a Dictionary - association word to numeric id
    regression_dictionary = corpora.Dictionary(tokenized_data)

    # later on, load trained model from file
    lda_regression_model =  models.LdaModel.load('regression_lda_model')
    myTopicsList = []

    for idx, topic in lda_regression_model.print_topics(5):
        myTopicsList.append(re.findall(r"\"(.*?)\"",lda_regression_model.print_topic(idx, 5)))

    searchSVMregression = "support vector SVR SVM maximal margin nonlinear kernel feature space dimensionality soft epsilon multipliers"
    searchlinearRegression = "mean root square correlation slope line error linear equation gradient descent average scatterplot bias"

    linearregressionCount = 0
    svmregressionCount = 0

    bow = regression_dictionary.doc2bow(relevant_words)
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
