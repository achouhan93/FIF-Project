# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:24:02 2019

@author: ashis
"""
from NLP_PreProcessing import nlp_pre_process
from Database_Processing import database_processing
from Comparison_Processing import comparison_values
from import_library import *

def user_story_processing(user_story):
    
    # NLP Pre-Processing 
    tokenize_words = nlp_pre_process.tokenize(user_story)
    punctuation_removed = nlp_pre_process.remove_punctuation(tokenize_words)
    stop_words_removed = nlp_pre_process.remove_stop_words(punctuation_removed)
    
    # Insights from Database
    server_connection = database_processing.mysql_connection('root','Ashish@123456789','localhost')
    databases_present = database_processing.database_information(server_connection)
    
    # Finding the Database to be referred
    database_finalised = comparison_values.processing_array_generation(stop_words_removed,databases_present)
    while(True):
        user_decision = input("Database Predicted by System is " + database_finalised.upper() + ". Is the prediction Correct?(Yes/No)")    
        if user_decision == "Yes":
            break
        elif user_decision == "No":
            print("Following are the list of Database Present:")
            count = 1
            for x in range(0, len(databases_present)):
                print(str(count) + " " + databases_present[x].upper())
                count = count + 1
            database_finalised = input("Enter the Correct Database Name: ").lower()
            break
        else:
            print("Kindly insert input in Yes or No")    

    fields, field_comments = database_processing.database_metadata_information(server_connection,database_finalised)
    
    idx = 0
    
    for field in fields:
        field = re.sub('[^0-9a-zA-Z]+', ' ', field)
        fields[idx] = field
        idx = idx + 1
        
    # Advance NLP Processing
    pos_tagged_words = nlp_pre_process.part_of_speech_tagging(stop_words_removed)  
    
    important_words = nlp_pre_process.important_words_extraction(pos_tagged_words)
    
    synonyms_values = nlp_pre_process.synonyms_words(important_words)
    
    if field_comments.size:
        relevant_columns_based_on_comments = comparison_values.processing_array_generation(field_comments,synonyms_values)
    
    if fields.size:
        relevant_columns_based_on_fields = comparison_values.processing_array_generation(fields,synonyms_values)
    
    
    
if __name__ == "__main__":
    user_story = input('Enter a User Story: ')
    user_story_processing(user_story.lower())