#!/usr/bin/python

from geopy.geocoders import Nominatim


def get_zipcode(addr_string):
    geolocator = Nominatim()
    location = geolocator.geocode(addr_string, addressdetails=True)
    return location.raw['address']['postcode'];

def get_zipcode(latlong):
    geolocator = Nominatim()
    location = geolocator.reverse(latlong)
    return location.raw['address']['postcode'];

with open('sample.csv','rb') as csvin, open('new.csv', 'wb') as csvout:
    for row in csvin:
        split_row = row
        split_row = row.split('\"')
        split_row_c = row.split(",")

        if len(split_row_c) > 10 and len(split_row) > 1:
            applicationDate = split_row_c[10]
            if "/" in applicationDate and applicationDate[-1].isdigit():
                applicationYear = applicationDate[-4:]
                latlong = split_row[-2][1:-1]
                latlong_split = latlong.split(",")
                if len(latlong_split) > 3:
                    latlong = latlong_split[-2] + "," + latlong_split[-1]
                zip = get_zipcode(latlong)
                line = ",".join([applicationYear, zip, row])

                csvout.write(line)
