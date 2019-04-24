# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:04:41 2019

@author: ashis
"""
from import_library import *
from NLP_PreProcessing import nlp_pre_process

data = pd.read_excel('FIF Dataset.xlsx')
data_text = data[['Details']]
# data_text['index'] = data_text.index
documents = data_text

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

for x in range(19):
    print("-----------------------")
    print("LDA Model: " + str(x))
    print("------------------------")
    
    #Running LDA Bag of Words
    lda_model = models.LdaModel(corpus=corpus, num_topics=2, id2word=dictionary, passes=20)
    
    file_name = 'lda_model' + str(x)
    
    lda_model.save(file_name)
    
    for idx, topic in lda_model.print_topics(5):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    lda_model.clear()
    

