#!/usr/bin/python

from __future__ import print_function
from geopy.geocoders import Nominatim


def get_zipcode(addr_string):
    geolocator = Nominatim()
    location = geolocator.geocode(addr_string, addressdetails=True)
    return location.raw['address']['postcode'];

def get_zipcode(latlong_string):
    geolocator = Nominatim()
    location = geolocator.reverse(latlong)
    return location.raw['address']['postcode'];

def get_zipcode_citystate(citystate_string):
    citystate_string = citystate_string.replace('\"', "")
    citystate_string = citystate_string.replace('\n', "")
    return citystate_string.split('-')[0]


with open('sample.csv','rb') as csvin, open('new.csv', 'wb') as csvout:
    headers = csvin.read()
    print(headers)
    for row in csvin:
        split_row = row.split(',')[-1].split(" ")
        if len(split_row) > 1:
            zip = get_zipcode_citystate(split_row[-1])
            #print(zip)

            line = zip + "," + row
##            print(line, file=csvout)
#            csvout.write(line)

addr_string = "Seattle, WA 98118"

geolocator = Nominatim()
location = geolocator.geocode(addr_string, addressdetails=True)

#location = geolocator.reverse("52.509669, 13.376294")

#print((location.latitude, location.longitude))
#print (location.raw['address']['postcode'])
