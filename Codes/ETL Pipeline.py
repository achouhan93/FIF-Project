# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 00:30:45 2018

@author: Ashish Chouhan
ETL Pipeline
"""

import google.cloud.dataflow as df
import argparse
import sys

def parse_record(e):
    import json
    r = json.loads(e)
    return r['ProductID'], r['Price']

def run():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    known_args, pipeline_args = parser.parser_known_args(sys.argv)
    
    p = df.Pipeline('DirectPipelineRunner')
    pc = p | df.io.Read(df.io.TextFileSource(known_args))
    df.FlatMap(parse_record)
    df.CombinePerKey(sum) 
    
    pc | df.io.Write(df.io.BigQuerySink(known_args.output, schema='ProductID:INTEGER, Value:FLOAT'))
 #   pc | df.io.Write(df.io.TextFileSink(known_args.output))
    p.run()
    
run()


