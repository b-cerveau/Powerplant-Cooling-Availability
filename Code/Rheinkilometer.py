import json
import csv
import pandas as pd

from shapely.geometry import Point, LineString
from pyproj import Transformer

###########################################################################################################
##################### Input filepaths

# Source : https://geoportal.bafg.de/arcgis3/rest/services/IKSR_Biotopverbund_2020/Rheinabschnitte/MapServer/0
RHEINKILOMETER = 'QGIS/Shapefiles/Rheinkilometer.geojson'
CLEANED_RHEINKILOMETER = 'QGIS/Shapefiles/rheinkm_cleaned.csv'

#################################################################################################################

###########################################
####### Initial kilometric data preprocessing
def cleanRheinkilometer():
    with open(RHEINKILOMETER, 'r') as f:
        data = json.load(f)

    # Only keep FLUSS == "2"
    filtered_points = []
    for feature in data['features']:
        if feature['properties'].get('FLUSS') == '2':
            lon, lat = feature['geometry']['coordinates']
            filtered_points.append((lat, lon))

    # Save in CSV file
    with open(CLEANED_RHEINKILOMETER, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['km_id', 'latitude', 'longitude'])
        for i, (lat, lon) in enumerate(filtered_points):
            writer.writerow([i, lat, lon])

###########################################
#######  Tool to get river kilometer from decimal coordinates 

points_latlon = []
with open(CLEANED_RHEINKILOMETER, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        points_latlon.append((lon, lat))  # x=lon, y=lat

# Project to UTM for accurate distances
transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
points_proj = [transformer.transform(lon, lat) for lon, lat in points_latlon]

# Create a list of cumulative distances and points
cumulative_distances = [0.0]
for i in range(1, len(points_proj)):
    p1 = Point(points_proj[i - 1])
    p2 = Point(points_proj[i])
    dist = p1.distance(p2)
    cumulative_distances.append(cumulative_distances[-1] + dist)

# Total river length in projected meters
total_length = cumulative_distances[-1]

# Create LineString for projection
river_line = LineString(points_proj)

def get_rhine_km_id(lat, lon):
    ''' Returns km id on the Rhine for a given point in (lat,lon) decimal coordinates. 
    
    km_id is as defined in the above rheinkilometer.geojson shapefile, and as such it : 
    starts at 1 near the Bodensee and increments up when moving downstream -> not the conventional km measure on a river.'''
    # Project input point
    pt_proj = Point(transformer.transform(lon, lat))
    # Get projected distance along the river line
    proj_dist = river_line.project(pt_proj)

    # Find which segment this projection falls in
    for i in range(1, len(cumulative_distances)):
        if cumulative_distances[i] >= proj_dist:
            # Interpolate km_id between i-1 and i
            d0 = cumulative_distances[i - 1]
            d1 = cumulative_distances[i]
            if d1 == d0:
                frac = 0
            else:
                frac = (proj_dist - d0) / (d1 - d0)
            km_id = (i - 1) + frac
            return km_id
    return len(cumulative_distances) - 1  # Last point fallback


