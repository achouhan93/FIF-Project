# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:24:02 2019

@author: ashis

If Date Column is present in the database then for User Story, date column cannot be retreived as features
"""
from import_library import *

# Database Information Retreival Service
from Database_Information_Retreival import database_processing

# Topic Modelling Service
from Topic_Modeller import lda_supervised_topic_modelling

# Natural Language Processing Services
from Tokenizer import tokenize
from Spelling_Corrector import spell_checker
from Stop_Words_Removal import remove_stop_words
from Synonymization import synonyms_words
from Part_of_Speech_Tagging import part_of_speech_tagging

# Word Embeddings Service
from Word_Embedding import word_embedding_tfidf
from Euclidean_Distance import euclidean_distance
from Manhattan_Distance import manhattan_distance
from Cosine_Similarity import cosine_similarity

# Feature Selection Orchestrator
from Feature_Identifier_Orchestrator import feature_selection_processing

# Data Mining Task Evalutaor Orchestrator
from Data_Mining_Task_Evaluator_Orchestrator import algorithm_selection_processing

from Counter_Processing import processing_array_generated

warnings.filterwarnings("ignore", category=FutureWarning)

def user_story_processing(user_story):
    
    existing_comparison_technique = ['cosine', 'euclidean' , 'manhattan']

    # NLP Pre-Processing 
    tokenize_words = tokenize(user_story)
    corrected_words = spell_checker(tokenize_words)
    stop_words_removed = remove_stop_words(corrected_words)
    hypothesis_synonyms_values = synonyms_words(stop_words_removed)
    
    lda_output = lda_supervised_topic_modelling(stop_words_removed)
    
    # Insights from Database
    server_connection = database_processing.mysql_connection('root','Ashish@123456789','localhost')
    databases_present = database_processing.database_information(server_connection)
    number_of_values = 1
    
    database_finalisation_list = []
    
    vectorized_words = word_embedding_tfidf(databases_present, hypothesis_synonyms_values)
    
    for comparison_technique in existing_comparison_technique:
        # Finding the Database to be referred
        if comparison_technique == "euclidean":
            extracted_database_finalised = euclidean_distance(databases_present, vectorized_words, number_of_values)
            database_finalisation_list.append(extracted_database_finalised)
        elif comparison_technique == "cosine":
            extracted_database_finalised = cosine_similarity(databases_present,vectorized_words, number_of_values)
            database_finalisation_list.append(extracted_database_finalised)
        elif comparison_technique == "manhattan":
            extracted_database_finalised = manhattan_distance(databases_present,vectorized_words, number_of_values)
            database_finalisation_list.append(extracted_database_finalised)
    
    database_finalised_value = processing_array_generated(database_finalisation_list, number_of_values)  
    database_finalised = database_finalised_value[0]
    
    while(True):
        user_decision = input("Database Predicted by System is " + database_finalised.upper() + ".\nIs the prediction Correct?\nYes - If Prediction is Correct\nNo - If Prediction is Wrong\nNA - Not Aware of Database\nq - To go Back : ")    
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
        elif user_decision == "NA":
            print("All Databases present in the Database Connection will be Considered")
            database_finalised = " "
            break
        elif user_decision == "q":
            return
        else:
            print("Kindly insert input in Yes or No")    
    
    database_metadata_information = []
    database_value = []
    table_information = []
    fields = []
    field_datatype = []
    field_comments = []

    if database_finalised == " ":
        for x in range(0, len(databases_present)):
            database_metadata_info, database_val, table_info, field_info , field_datatype_info , field_comments_info = database_processing.database_metadata_information(server_connection,databases_present[x])
            database_metadata_information.extend(database_metadata_info)
            database_value.extend(database_val)
            table_information.extend(table_info)
            fields.extend(field_info)
            field_datatype_info.extend(field_datatype)
            field_comments.extend(field_comments_info)
            
    else:
        database_metadata_information, database_value, table_information, fields, field_datatype, field_comments = database_processing.database_metadata_information(server_connection,database_finalised)
    
    updated_fields_complete = []
    
    for field in fields:
        field = re.sub('[^0-9a-zA-Z]+', ' ', field)
        updated_fields_complete.append(field)
    
    updated_fields = pd.unique(updated_fields_complete).tolist()
    field_comments = pd.unique(field_comments).tolist()
        
    # Advance NLP Processing
    #relevant_words = [words for words in stop_words_removed if len(words) > 3]
    pos_tagged_words = part_of_speech_tagging(stop_words_removed)       
    synonyms_values = synonyms_words(pos_tagged_words)

    if (len(updated_fields) <= pos_tagged_words.size):
        number_of_values = len(updated_fields)
    else:
        number_of_values = pos_tagged_words.size
    
    # Field Value Processing
    relevant_columns_based_on_comments = []
    relevant_columns_based_on_fields = []
    
    column_predicted_list = []
    
    if len(updated_fields):
        vectorized_field_words = word_embedding_tfidf(updated_fields, synonyms_values)
        
        for comparison_technique in existing_comparison_technique:
            # Finding the Database to be referred
            if comparison_technique == "euclidean":
                relevant_columns_based_on_fields = euclidean_distance(updated_fields, vectorized_field_words, number_of_values)
            elif comparison_technique == "cosine":
                relevant_columns_based_on_fields = cosine_similarity(updated_fields, vectorized_field_words, number_of_values)
            elif comparison_technique == "manhattan":
                relevant_columns_based_on_fields = manhattan_distance(updated_fields, vectorized_field_words, number_of_values)
            
            column_predicted_list.extend(relevant_columns_based_on_fields)
   
    if (len(field_comments) and len(updated_fields) == len(field_comments)):
        vectorized_comment_words = word_embedding_tfidf(field_comments, synonyms_values)
        
        for comparison_technique in existing_comparison_technique:
            # Finding the Database to be referred
            if comparison_technique == "euclidean":
                relevant_columns_based_on_comments = euclidean_distance(field_comments, vectorized_comment_words, number_of_values)
            elif comparison_technique == "cosine":
                relevant_columns_based_on_comments = cosine_similarity(field_comments, vectorized_comment_words, number_of_values)
            elif comparison_technique == "manhattan":
                relevant_columns_based_on_comments = manhattan_distance(field_comments, vectorized_comment_words, number_of_values)

            relevant_fields_based_on_comments = []
            
            for comments in relevant_columns_based_on_comments:
                relevant_fields_based_on_comments.append(updated_fields[field_comments.index(comments)])
            
            column_predicted_list.extend(relevant_fields_based_on_comments)
    
    number_of_values = len(list(set(column_predicted_list)))       
    column_finalised = processing_array_generated(column_predicted_list, number_of_values)
    
    field_finalised = []
    
    for field_value in column_finalised:
        field_finalised.append(fields[updated_fields_complete.index(field_value)])
    
    finalised_database = []
    finalised_table = []
    
    for field in field_finalised:
        indices = [i for i, x in enumerate(fields) if x == field]
        field_database = []
        field_table = []
        index = 0
        for z in indices:
            field_database.insert(index,database_value[z].upper())
            field_table.insert(index,table_information[z].upper())
            index = index + 1
        
        field_database = pd.unique(field_database).tolist()
        field_table = pd.unique(field_table).tolist()
        finalised_database.append(field_database)
        finalised_table.append(field_table)
        
    print('**** After NLP Processing ****')
    result_display(field_finalised, finalised_table, finalised_database)
    
    print('**** After Feature Selection ****')
    field_finalised, finalised_table, finalised_database, feature_list, logs, feature_encoded = feature_selection_processing(field_finalised, finalised_table, finalised_database, server_connection)
    print('**** Logs ****')
    for x in range(len(logs)):
        print(logs[x])
    result_display(field_finalised, finalised_table, finalised_database) 
    
    if (lda_output[0] != " ") and (len(field_finalised) != 0):
        print('**** Probable Algorithms ****')
        algorithm_used, accuracy_score, target_feature, independent_features, message = algorithm_selection_processing(feature_list, lda_output, feature_encoded)
        
        if message == " ":
            table = PrettyTable(['Preferences' , 'Algorithm Prefered','Accuracy Percentage', 'Target Feature (Field Name__Table Name__Database Name)', 'Independent Features'])
            index = 1
            for i in range(len(algorithm_used)):
                table.add_row([index, algorithm_used[index-1], accuracy_score[index-1], target_feature[index-1], independent_features[index-1]])
                index = index + 1
                
            print(table)
        else:
            print(message)
            
def result_display(finalised_field, tables, database):
    table = PrettyTable(['Preferences','Field Name', 'Tables', 'Database'])
    index = 1
    for field in finalised_field:
        table.add_row([index, field, tables[index-1] , database[index-1]])
        index = index + 1
        
    print(table)
    
if __name__ == "__main__":
    while(True):
        user_story = input('Enter a User Story or Press "q" to exit the application: ')
        if user_story == " ":
            print("Kindly insert a User Story")
            continue
        elif user_story == "q":
            break
        else:
            user_story_processing(user_story)
        