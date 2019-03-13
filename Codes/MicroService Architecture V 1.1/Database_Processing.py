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
        login = args[0]
        password = args[1]
        host = args[2]
        connection = MySQLdb.connect (user = login,
                                      passwd=password,
                                      host=host)

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