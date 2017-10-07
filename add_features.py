# -*- coding: utf-8 -*-

import config
import model_stats

import os, sys, yaml, logging
import time

import numpy as np
import pandas as pd

def get_zipcode(long, lat):
    return 98052;

def add_features(df, year, zipcode):

    df['zipcode'] = get_zipcode(df['long'], df['lat'])
    
    return df

   
def main():

    start_time = time.time()

    """Reads configuration and adds features to the raw data."""
    
    options = config.get_config()
    logging.debug('Reading input %s' % options.input)
    df = pd.read_pickle(options.input)
    logging.debug('Finished reading %s' % options.input)

    df = add_features(df)
    df.to_pickle(options.processed)
    
    print("--- Execution time:  %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()

