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
        elif len(kwargs) == 4 or len(kwargs) == 4:
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
        sql_query = "SHOW DATABASES"
        cursor= conn_obj.cursor()
        cursor.execute(sql_query)
        database_present = cursor.fetchall()
        database_array = [w[0].lower() for w in database_present]
        return database_array
    
    def database_metadata_information(conn_obj,database):
        # Retreive Information About Database Metadata Present in the Connection
        database_metadata = "SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = '"+ database + "'"
        cursor= conn_obj.cursor()
        cursor.execute(database_metadata)
        database_metadata_information = cursor.fetchall()
        fields = [x[3] for x in database_metadata_information]
        field_comments = [x[19] for x in database_metadata_information]
        return (fields, field_comments)