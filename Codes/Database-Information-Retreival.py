# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:16:38 2019
@author: Ashish Chouhan
Description:
    RToS Implementation without LDA but all Algorithm Covered
    (Micro Service Architecture)
"""
from import_library import *

class database_processing():

    def mysql_connection(*args,**kwargs):
        # MySQL Connection
        if len(args) == 3 or len(kwargs) == 3:
            if len(args) == 3:
                login = args[0]
                password = args[1]
                host = args[2]
            elif len(kwargs) == 3:
                login = kwargs['login']
                password = kwargs['password']
                host = kwargs['host']

            connection = MySQLdb.connect (user = login,
                                                  passwd=password,
                                                  host=host)
            
        elif len(args) == 4 or len(kwargs) == 4:
            if len(args) == 4:
                login = args[0]
                password = args[1]
                host = args[2]
                database = args[3]
            elif len(kwargs) == 4:
                login = kwargs['login']
                password = kwargs['password']
                host = kwargs['host']
                database = kwargs['database']

            connection = MySQLdb.connect (user = login,
                                                  passwd = password,
                                                  host = host,
                                                  database = database)

        return connection


    def database_information(conn_obj):
        # Retreive Information About Databases Present in the Connection
        sql_query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema','performance_schema','mysql','sys')"
        cursor = conn_obj.cursor()
        cursor.execute(sql_query)
        database_present = cursor.fetchall()
        database_array = [w[0].lower() for w in database_present]
        return database_array
    
    def database_metadata_information(conn_obj,database):
        # Retreive Information About Database Metadata Present in the Connection
        database_metadata = "SELECT table_schema, table_name, column_name, data_type, column_comment FROM information_schema.columns WHERE table_schema = '"+ database + "'"
        cursor= conn_obj.cursor()
        cursor.execute(database_metadata)
        database_metadata_information = cursor.fetchall()
        database_value = [x[0] for x in database_metadata_information]
        table_information = [x[1] for x in database_metadata_information]
        fields = [x[2] for x in database_metadata_information]
        field_datatype = [x[3] for x in database_metadata_information]
        field_comments = [x[4] for x in database_metadata_information]
        return (database_metadata_information, database_value, table_information, fields, field_datatype, field_comments)
    
    def table_information(database, table, conn_obj):
        
        try:
            # Retreive Information About Database Metadata Present in the Connection
            table_data = "SELECT * FROM " + database + "." + table
            cursor= conn_obj.cursor()
        
            cursor.execute(table_data)
            table_data_information = cursor.fetchall()
        
            table_metadata = "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = '" + database + "' and table_name = '" + table + "'"
            cursor.execute(table_metadata)
            table_metadata_information = cursor.fetchall()
            column_values = [x[2] for x in table_metadata_information]
            
            error_code = "200"
        except:
            table_data_information = []
            column_values = []
            table_metadata_information = []
            error_code = "404"
        
        return (pd.DataFrame(list(table_data_information), columns = column_values), pd.DataFrame(list(table_metadata_information)), error_code)
    