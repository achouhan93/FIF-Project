# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:25:52 2019

@author: Ashish
"""
from import_library import *
from Database_Information_Retreival import database_processing
from Feature_Scaler import scaling
from Duplicate_Feature_Removal import duplicate_removal
from Encoding import label_encoding
from Variance_Threshold import variance_feature
from Correlation_Analysis import feature_correlation

def feature_selection_processing(database_fields, database_tables, database_list, database_connection):
    
    feature_encoded = []
    logger = []
    relevant_columns_data = pd.DataFrame()
    
    for field in database_fields:
        indices = database_fields.index(field)
        databases = database_list[indices]
        tables = database_tables[indices]
        
        for database in databases:
            for table in tables:
                # Fetch the relevant table and its metadata information
                table_data, table_metadata, error_code = database_processing.table_information(database, table, database_connection)
                
                if (error_code == "404"):
                    continue
                
                if table_data.empty:
                    log = table + " in " + database + " is empty."
                    logger.append(log)
                    continue
                    
                # Encoding Categorical Data
                for column_name in table_data.columns:
                    if table_data[column_name].dtype == object:
                        table_data[column_name] = label_encoding(table_data[column_name])
                        table_data[column_name].fillna(0, inplace=True)
                        feature_encoded.append((column_name, table, database))
                    else:
                        pass
                
                table_data = table_data.select_dtypes(include=np.number)
                table_data.fillna(table_data.mean(), inplace=True)
                
                try:
                    df_Y = table_data.loc[:, field].to_frame(name = field)
                    df_X = table_data.drop(field, axis = 1)
                except:
                    continue
                
                initial_features = df_X
                (df_X, logger) = variance_feature(df_X, logger)
                df_X = duplicate_removal(df_X)
                
                duplicated_columns = [dup_col for dup_col in initial_features.columns if dup_col not in df_X.columns]
    
                if duplicated_columns:
                    log = (",".join(duplicated_columns)) + " : Duplicate Columns are removed"
                    logger.append(log) 
                
                # Scaling the Features on same scale
                df_X_scaled = scaling(df_X)
                df_Y_scaled = scaling(df_Y)
                number_of_features_relevant = round(len(df_X_scaled.columns)/2)
                features, features_data = feature_correlation(df_X_scaled, df_Y_scaled, number_of_features_relevant)
                
                for value in range(len(features_data.columns)):
                    features_data.columns.values[value] = features[value] + '___' + table + '___' + database
                    
                relevant_columns_data.reset_index(drop=True, inplace=True)
                features_data.reset_index(drop=True, inplace=True)
                relevant_columns_data = pd.concat([relevant_columns_data, features_data], axis = 1)
    
    final_fields = []
    final_table = []
    final_database = []
    
    logger = list(sorted(set(logger)))
    
    if relevant_columns_data.empty:
        return (final_fields, final_table, final_database, relevant_columns_data, logger, feature_encoded)
            
    relevant_columns_data = duplicate_removal(relevant_columns_data)
 
    for i in range(len(relevant_columns_data.columns)):
        field_value, table_value, database_value = relevant_columns_data.columns[i].split("___")
        final_fields.append(field_value)
        final_table.append(table_value)
        final_database.append(database_value)
        
    return (final_fields, final_table, final_database, relevant_columns_data, logger, feature_encoded)
    
