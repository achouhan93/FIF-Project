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
    
    existing_comparison_technique = ['cosine', 'euclidean' , 'manhattan' , 'cityblock']

    # NLP Pre-Processing 
    tokenize_words = nlp_pre_process.tokenize(user_story)
    punctuation_removed = nlp_pre_process.remove_punctuation(tokenize_words)
    stop_words_removed = nlp_pre_process.remove_stop_words(punctuation_removed)
    
    # Insights from Database
    server_connection = database_processing.mysql_connection('root','Ashish@123456789','localhost')
    databases_present = database_processing.database_information(server_connection)
    number_of_values = 1
    
    
    database_finalisation_list = []

    for comparison_technique in existing_comparison_technique:
        # Finding the Database to be referred
        extracted_database_finalised = comparison_values.similar_values(databases_present, stop_words_removed, number_of_values, comparison_technique)
        database_finalisation_list.append(extracted_database_finalised)
    
    database_finalised = comparison_values.processing_array_generated(database_finalisation_list, number_of_values)  
    
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
    
    updated_fields = []
    
    for field in fields:
        field = re.sub('[^0-9a-zA-Z]+', ' ', field)
        updated_fields.append(field)
        
    # Advance NLP Processing
    pos_tagged_words = nlp_pre_process.part_of_speech_tagging(stop_words_removed)  
    important_words = nlp_pre_process.important_words_extraction(pos_tagged_words)
    number_of_values = important_words.size
    
    synonyms_values = nlp_pre_process.synonyms_words(important_words)
    
    # Field Value Processing
    relevant_columns_based_on_comments = []
    relevant_columns_based_on_fields = []
    
    column_predicted_list = []
    
    if len(updated_fields):
        for comparison_technique_present in existing_comparison_technique:
            relevant_columns_based_on_fields = comparison_values.similar_values(updated_fields, synonyms_values, number_of_values, comparison_technique_present)
            column_predicted_list.extend(relevant_columns_based_on_fields)
   
    if len(field_comments):
        for comparison_technique in existing_comparison_technique:
            relevant_columns_based_on_comments = comparison_values.similar_values(field_comments, synonyms_values, number_of_values, comparison_technique)
            relevant_fields_based_on_comments = []
            
            for comments in relevant_columns_based_on_comments:
                relevant_fields_based_on_comments.append(updated_fields[field_comments.index(comments)])
            
            column_predicted_list.extend(relevant_fields_based_on_comments)
            
    column_finalised = comparison_values.processing_array_generated(column_predicted_list, number_of_values)
    
    field_finalised = []
    
    for field_value in column_finalised:
        field_finalised.append(fields[updated_fields.index(field_value)])
    
    print('**** After NLP Processing ****')
    result_display(field_finalised, database_finalised)
    #most_relevant_features = feature_selection_script.feature_selection_processing(field_finalised, server_connection)
    
    #print(most_relevant_features)

def result_display(finalised_field, database):
    table = PrettyTable(['Field Name', 'Database'])
    if database != ' ':
        for field in finalised_field:
            table.add_row([field, database.upper()])
        
        print(table)
    
if __name__ == "__main__":
    user_story = input('Enter a User Story: ')
    #user_story_processing(user_story.lower())
    user_story_processing(user_story)