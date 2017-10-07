#!/usr/bin/python

# -*- coding: utf-8 -*-

import config

import os, sys, yaml, logging
import time

import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim

def get_zipcode(addr_string):
    geolocator = Nominatim()
    location = geolocator.geocode(addr_string, addressdetails=True)
    return location.raw['address']['postcode'];

def get_zipcode(long, lat):
    geolocator = Nominatim()
    location = geolocator.reverse(lat, long)
    return location.raw['address']['postcode'];

def add_features(df, year, zipcode):
    df['zipcode'] = get_zipcode(df['lat'], df['long'])
    return df

def __main__():
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
