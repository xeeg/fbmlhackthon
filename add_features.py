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


def get_zip(string):
   city,zipcode = string.split(',')
   if city == 'SEATTLE':
       a,b,c = zipcode.split(" ")
       d,e = c.split('-')
       print(d)
       zipcode = d
       return d
   else:
       return 0
                               
def clean_up_zipcode(infile, outfile):
    
               

finput_older = "/Users/homaei/Documents/Projects/FBMLHackthon2017/fbmlhackthon/datasets/raw/"
files= {2011: '2011_Active_Business_License_Data.csv',
        2012: '2012_Business_License_Data__Updated_June_2013_.csv',
        2013: '2013_Active_Business_License_Data.csv',
        2014: '2014_Active_Business_License_Data.csv',
        2015: '2015_Active_Business_License_Data.csv',
        2016: '2016_Active_Businesses_License_Data.csv'}

col_names = {2011: 'City, State, Zip',
            2012: 'city_state_zip',
            2013: 'City, State and ZIP',
            2014: 'City, State and ZIP',
            2015: 'City, State, ZIP',
            2016: 'City, State, Zip'}

dx = pd.DataFrame()
for i, year in enumerate(files):
    print(folder+files[year])
    col_name = col_names[year]
    df = pd.read_csv(folder+files[year])  
    # Filter for Seaattle 
    df = df[df[col_name].str.split('-').str[0].str.split(',').str[0] == 'SEATTLE' ]
    df["zipcode"] = df[col_name].str.split('-').str[0].str.split("SEATTLE, WA").str[1]
    df["year"] = year
    dx = pd.append(df)

df = dx
df["yearzip"] = df["year"].astype(str) + df["zipcode"].astype(str) 
cols = ["yearzip", "business_id"]
df[cols]
df_count = df[cols].groupby("yearzip").count()
    


output_file = "building_perms_current_cleaned.csv"


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
