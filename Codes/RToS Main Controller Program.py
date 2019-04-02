# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:24:02 2019

@author: ashis
"""
from import_library import *
from NLP_PreProcessing import nlp_pre_process
from Database_Processing import database_processing
from Comparison_Processing import comparison_values
from feature_selection import feature_selection_processing

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
    
    updated_fields = []
    
    for field in fields:
        field = re.sub('[^0-9a-zA-Z]+', ' ', field)
        updated_fields.append(field)
    
    updated_fields = pd.unique(updated_fields).tolist()
    field_comments = pd.unique(field_comments).tolist()
        
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
    
    most_relevant_features = feature_selection_processing(field_finalised, finalised_table, finalised_database, server_connection)
    

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
        if user_story == "":
            print("Kindly insert a User Story")
            continue
        elif user_story == "q":
            break
        else:
            user_story_processing(user_story)
        