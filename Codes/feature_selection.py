# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:25:52 2019

@author: Ashish
"""
from import_library import *
from Database_Processing import database_processing

def feature_selection_processing(database_fields, database_tables, database_list, database_connection):
    
    lda_result = ' '
    
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
                
                # Encoding Categorical Data
                for column_name in table_data.columns:
                    if table_data[column_name].dtype == object:
                        labelencoder = LabelEncoder()
                        table_data[column_name] = labelencoder.fit_transform(table_data[column_name].astype(str))
                    else:
                        pass
                
                table_data = table_data.select_dtypes(include=np.number)
                table_data.fillna(table_data.mean(), inplace=True)
                
                try:
                    df_Y = table_data.loc[:, field].to_frame(name = field)
                    df_X = table_data.drop(field, axis = 1)
                except:
                    continue
                
                df_X = filter_method_data_preprocessing(df_X)
                
                # Scaling the Features on same scale
                sc_x = StandardScaler()
                df_X_scaled = sc_x.fit_transform(df_X)
                df_X_scaled = pd.DataFrame(df_X_scaled, index = df_X.index, columns = df_X.columns)
                sc_y = StandardScaler()
                df_Y_scaled = sc_y.fit_transform(df_Y)
                df_Y_scaled = pd.DataFrame(df_Y_scaled, index = df_Y.index, columns = df_Y.columns)
                
                number_of_features_relevant = round(len(df_X_scaled.columns)/2)
    
                features, features_data = filter_method_execution(df_X, df_Y, number_of_features_relevant)
                
                for value in range(len(features_data.columns)):
                    features_data.columns.values[value] = features[value] + '___' + table + '___' + database
                    
                relevant_columns_data.reset_index(drop=True, inplace=True)
                features_data.reset_index(drop=True, inplace=True)
                relevant_columns_data = pd.concat([relevant_columns_data, features_data], axis = 1)
    
    relevant_columns_data_T = relevant_columns_data.T
    relevant_columns_data = relevant_columns_data_T.drop_duplicates(keep='first').T
    
    """
    if (lda_result == " "):
        if len(relevant_columns_data.columns) > len(database_fields):
            correlated_features = set()  
            correlation_matrix = relevant_columns_data.corr()
            
            # Extract Co-Related Columns
            for i in range(len(correlation_matrix.columns)):  
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        colname = correlation_matrix.columns[i]
                        correlated_features.add(colname)
            
        else:
            return relevant_columns_data
    else:
        relevant_features = filter_advance_method(df_X, df_Y, number_of_features_relevant)
    """
    
    final_fields = []
    final_table = []
    final_database = []
    
    if (lda_result == " "):
        
        for i in range(len(relevant_columns_data.columns)):
            field_value, table_value, database_value = relevant_columns_data.columns[i].split("___")
            final_fields.append(field_value)
            final_table.append(table_value)
            final_database.append(database_value)
        
        return (final_fields, final_table, final_database)
    
    else:
        advanced_columns_data = filter_advance_method(relevant_columns_data, lda_result, len(database_fields))
        
        for i in range(len(advanced_columns_data)):
            field_value, table_value, database_value = advanced_columns_data.split("___")
            final_fields.append(field_value)
            final_table.append(table_value)
            final_database.append(database_value)
        
        return (final_fields, final_table, final_database)
        

def filter_method_data_preprocessing(X_dataframe):
    
    # Constant Columns Removal
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_dataframe)       
    
    # Remove the constant columns
    constant_columns = [column for column in X_dataframe.columns  
                    if column not in X_dataframe.columns[constant_filter.get_support()]]
    X_dataframe.drop(labels=constant_columns, axis=1, inplace=True)
        
    # Quasi-Constant Columns Removal
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(X_dataframe)
    # Remove the Quasi-Constant Columns
    qconstant_columns = [column for column in X_dataframe.columns  
                    if column not in X_dataframe.columns[qconstant_filter.get_support()]]
    X_dataframe.drop(labels=qconstant_columns, axis=1, inplace=True)
        
    #Duplicate Removal
    X_dataframe_T = X_dataframe.T
    X_dataframe = X_dataframe_T.drop_duplicates(keep='first').T
        
    return X_dataframe
    
def filter_method_execution(X_features, Y_target, no_of_features):
    
    # ---------------------------------------------------------------------------------- #        
    correlated_features = []
    
    X_features.reset_index(drop=True, inplace=True)
    Y_target.reset_index(drop=True, inplace=True)
    complete_data = pd.concat([X_features, Y_target], axis = 1)    
    correlation_matrix = complete_data.corr()
    
    # Extract Correlated features
    correlation_information = correlation_matrix[Y_target.columns[0]].sort_values(ascending=False).head(no_of_features)
    
    for i in range(len(correlation_information)):
        if abs(correlation_information[i]) > 0.8:
            correlated_features.append(correlation_information.index[i])
    
    correlated_features_dataframe = complete_data.loc[:, correlated_features]
    
    return (correlated_features, correlated_features_dataframe)
    
def filter_advance_method(complete_dataset, lda_model_prediction, no_of_features):
    """# ----------------------------------------------------------------------------------- #

    # Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain
    # model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute

    # Feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, Y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))"""
    
    print("Feature Ranking: %s" % (fit.ranking_))


        
        
                
                
                
            