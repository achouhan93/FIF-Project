# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 00:25:52 2019

@author: Ashish
"""
from import_library import *
from Database_Processing import database_processing
from Comparison_Processing import comparison_values

def feature_selection_processing(database_fields, database_tables, database_list, database_connection):
    
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
                
                (df_X, logger) = filter_method_data_preprocessing(df_X, logger)
                
                # Scaling the Features on same scale
                sc_x = StandardScaler()
                df_X_scaled = sc_x.fit_transform(df_X)
                df_X_scaled = pd.DataFrame(df_X_scaled, index = df_X.index, columns = df_X.columns)
                sc_y = StandardScaler()
                df_Y_scaled = sc_y.fit_transform(df_Y)
                df_Y_scaled = pd.DataFrame(df_Y_scaled, index = df_Y.index, columns = df_Y.columns)
                
                number_of_features_relevant = round(len(df_X_scaled.columns)/2)
    
                features, features_data = filter_method_execution(df_X_scaled, df_Y_scaled, number_of_features_relevant)
                
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
        return (final_fields, final_table, final_database, relevant_columns_data, logger)
            
    relevant_columns_data.fillna(0, inplace=True)
    relevant_columns_data_T = relevant_columns_data.T
    relevant_columns_data = relevant_columns_data_T.drop_duplicates(keep='first').T
 
    for i in range(len(relevant_columns_data.columns)):
        field_value, table_value, database_value = relevant_columns_data.columns[i].split("___")
        final_fields.append(field_value)
        final_table.append(table_value)
        final_database.append(database_value)
        
    return (final_fields, final_table, final_database, relevant_columns_data, logger)
        

def filter_method_data_preprocessing(X_dataframe, logs):
    
    intial_X_dataframe = X_dataframe
    
    # Constant Columns Removal
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_dataframe)       
    
    # Remove the constant columns
    constant_columns = [column for column in X_dataframe.columns  
                    if column not in X_dataframe.columns[constant_filter.get_support()]]
    X_dataframe.drop(labels=constant_columns, axis=1, inplace=True)
    
    if constant_columns:
        log = (",".join(constant_columns)) + " : Constant Columns are removed"
        logs.append(log)
        
    # Quasi-Constant Columns Removal
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(X_dataframe)
    # Remove the Quasi-Constant Columns
    qconstant_columns = [column for column in X_dataframe.columns  
                    if column not in X_dataframe.columns[qconstant_filter.get_support()]]
    X_dataframe.drop(labels=qconstant_columns, axis=1, inplace=True)
    
    if qconstant_columns:
        log = (",".join(qconstant_columns)) + " : Quasi-Constant Columns are removed"
        logs.append(log)    
    
    #Duplicate Removal
    X_dataframe_T = X_dataframe.T
    X_dataframe = X_dataframe_T.drop_duplicates(keep='first').T
    
    duplicated_columns = [dup_col for dup_col in intial_X_dataframe.columns if dup_col not in X_dataframe.columns]
    
    if duplicated_columns:
        log = (",".join(duplicated_columns)) + " : Duplicate Columns are removed"
        logs.append(log)    
        
    return (X_dataframe, logs)
    
def filter_method_execution(X_features, Y_target, no_of_features):
    
    # ---------------------------------------------------------------------------------- #        
    correlated_features = []
    
    X_features.reset_index(drop=True, inplace=True)
    Y_target.reset_index(drop=True, inplace=True)
    complete_data = pd.concat([X_features, Y_target], axis = 1)    
    
    existing_correlation_technique = ['pearson', 'kendall' , 'spearman']
    
    for correlation_technique in existing_correlation_technique:
        correlation_matrix = complete_data.corr(method=correlation_technique)
    
        # Extract Correlated features
        correlation_information = correlation_matrix[Y_target.columns[0]].sort_values(ascending=False).head(no_of_features)
    
        for i in range(len(correlation_information)):
            if abs(correlation_information[i]) > 0.8:
                correlated_features.append(correlation_information.index[i])
    
    correlated_features_finalised = comparison_values.processing_array_generated(correlated_features, len(X_features.columns))      
    correlated_features_dataframe = complete_data.loc[:, correlated_features_finalised]
    
    return (correlated_features, correlated_features_dataframe)
    
def algorithm_selection_processing(relevant_columns, lda_output):
    
    algorithm_details = []
    
    for i in range(len(relevant_columns.columns)):
        column_data = relevant_columns.copy()
        train_features, test_features, train_labels, test_labels = train_test_split(  
                    column_data.drop(labels=[column_data.columns[i]], axis=1),
                    column_data[column_data.columns[i]],
                    test_size=0.2,
                    random_state=41)
        
        if lda_output[0] == "Regression":
            if lda_output[1] == "LinearRegression" or lda_output[1] != " ":
                accuracy_percent, target_columns, features_columns  = linear_regression_processing(train_features, test_features, train_labels, test_labels)
                details = ("Regression : Linear Regression", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
        elif lda_output[0] == "Classification":
            if lda_output[1] == "SVM":
                accuracy_percent, target_columns, features_columns  = SVM_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : SVM(Kernel - Linear)", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == "DecisionTree":
                accuracy_percent, target_columns, features_columns  = decision_tree_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Decision Tree", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == "RandomForest":
                accuracy_percent, target_columns, features_columns  = random_forest_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Random Forest", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
            elif lda_output[1] == " ":
                accuracy_percent, target_columns, features_columns  = SVM_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : SVM(Kernel - Linear)", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                """
                accuracy_percent, target_columns, features_columns  = knn_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : K-Nearest Neighbors", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = naive_bayes_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Naive Bayes", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                """
                accuracy_percent, target_columns, features_columns  = decision_tree_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Decision Tree", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
                accuracy_percent, target_columns, features_columns  = random_forest_processing(train_features, test_features, train_labels, test_labels)
                details = ("Classification : Random Forest", accuracy_percent, target_columns, features_columns)
                algorithm_details.append(details)
                
    algorithm_details = sorted(algorithm_details, key = lambda x: x[1], reverse = True)     

    algorithms = []
    accuracy_score = []
    target_values = []
    independent_values = []
    
    for i in range(len(algorithm_details)):
        algorithms.append(algorithm_details[i][0])
        accuracy_score.append(algorithm_details[i][1])
        target_values.append(algorithm_details[i][2])
        independent_values.append(algorithm_details[i][3])
        
    return (algorithms, accuracy_score, target_values, independent_values)
    
def linear_regression_processing(X_train, X_test, Y_train, Y_test):
    
    model = LinearRegression()
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = LinearRegression()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = r2_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())
    
def SVM_processing(X_train, X_test, Y_train, Y_test):
    
    model = SVR(kernel="linear")
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = SVR(kernel="linear")
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = r2_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())
        
"""def knn_processing(X_train, X_test, Y_train, Y_test):
    
    model = KNeighborsRegressor()
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = KNeighborsRegressor()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = roc_auc_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())        

def naive_bayes_processing(X_train, X_test, Y_train, Y_test):
    
    model = MultinomialNB()
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = MultinomialNB()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = roc_auc_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())                
 """               
def decision_tree_processing(X_train, X_test, Y_train, Y_test):
    
    model = DecisionTreeRegressor()
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = DecisionTreeRegressor()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = r2_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())                

def random_forest_processing(X_train, X_test, Y_train, Y_test):
    
    model = RandomForestRegressor()
    rfe = RFE(model, round(len(X_train.columns)/2))
    features = rfe.fit(X_train.values, Y_train.values)
    value_index = []
    for i in range(len(features.ranking_)):
        if (features.ranking_[i] == 1):
            value_index.append(i)
            
    filtered_features = X_train.columns[list(value_index)]
    
    testing_model = RandomForestRegressor()
    testing_model.fit(X_train[filtered_features], Y_train)
    Y_pred = testing_model.predict(X_test[filtered_features])
    
    model_accuracy = r2_score(Y_test.values, Y_pred)
    
    return (model_accuracy, pd.DataFrame(Y_train).columns[0], filtered_features.tolist())            