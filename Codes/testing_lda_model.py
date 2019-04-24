# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:17:05 2019

@author: ashis
"""
from import_library import *
from NLP_PreProcessing import nlp_pre_process

data = pd.read_excel('FIF Dataset.xlsx')
data_text = data[['Details']]
# data_text['index'] = data_text.index
documents = data_text

evaluation = pd.read_excel('FIF Evaluation.xlsx', header=0, skiprows=0)
user_story_list = evaluation[['User Story']]
user_story = user_story_list


def clean_text(text):
    # NLP Pre-Processing 
    tokenize_words = nlp_pre_process.tokenize(text.lower())
    punctuation_removed = nlp_pre_process.remove_punctuation(tokenize_words)
    stop_words_removed = nlp_pre_process.remove_stop_words(punctuation_removed)
    relevant_words = [WordNetLemmatizer().lemmatize(words, pos='v') for words in stop_words_removed if len(words) > 3]
    return relevant_words

# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for x in range(len(documents)):
    text = documents['Details'][x]
    tokenized_data.append(clean_text(text))

# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)

# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# later on, load trained model from file
lda_model =  models.LdaModel.load('lda_model11')
myTopicsList = []

for idx, topic in lda_model.print_topics(5):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    myTopicsList.append(re.findall(r"\"(.*?)\"",lda_model.print_topic(idx, 5)))

searchClassification = " classification discrete labels class category observation branch group separate decision sets divided segmentation segregate"
searchRegression = "regression linear analyze distribution predict statisticsapproximate continuous range quantity series forecasting estimate count"

for y in range(1, len(user_story)):
    regressionCount = 0
    classificationCount = 0

    text = user_story['User Story'][y]
    print("-----------------------------------")
    print("USER STORY {} : {} ".format(y,text))

    bow = dictionary.doc2bow(clean_text(text))
    ldaData = lda_model[bow]

    if ldaData[0][1] == ldaData[1][1]:
        print("Equally Distributed")
        print("-----------------------------------")
    else:
        print("User_Story Prediction : ", ldaData)
    
        sort = sorted(ldaData, key=lambda tup:(-tup[1], tup[0]))
        topic_num = (sort[0][0])
        topics = lda_model.print_topic(topic_num, 5)
    
        print("Topic")
        topicKeywords =myTopicsList[topic_num]
        print(topicKeywords)
    
        for word in topicKeywords:
            if word in searchRegression:
                regressionCount+=1
            if word in searchClassification:
                classificationCount+=1
    
        print("regressionCount: {} , classificationCount: {}".format(regressionCount, classificationCount))
       
        if regressionCount > classificationCount:
            print("Regression")
        else:
            print("Classification")
        
        print("-----------------------------------")