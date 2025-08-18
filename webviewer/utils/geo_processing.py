from geoutils import load_country_raster_splitted, dualtree_on_rasters
import json
import pandas as pd

def load_countries():
    with open('data/countries.json') as f:
        countries = json.load(f)
    return countries

def process_country_data(country, capacity, population_threshold):
    rasters, transforms = load_country_raster_splitted(country, population_tresh=population_threshold)
    centers = dualtree_on_rasters(rasters, transforms, capacity=capacity)
    return centers
    