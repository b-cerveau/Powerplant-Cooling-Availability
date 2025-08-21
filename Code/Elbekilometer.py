import csv
from shapely.geometry import Point, LineString
from pyproj import Transformer

################################################## 
######### Input filepath

ELBE_CSV = "QGIS/Shapefiles/Elbekilometer.csv"

# Source : https://cdn.arcgis.com/home/item.html?id=180e184ef4a846a1a08b3d357cf9d309

##################################################

# Load East/North and river km
points_proj = []
elbe_km_values = []

with open(ELBE_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["East"] or not row["North"]:  # skip empty coords
            continue
        if row["Streckenbez"] == "Elbe_Aestur":  # skip estuary
            continue
        
        east = float(row["East"])
        north = float(row["North"])
        elbekm = float(row["elbeKM"])
        points_proj.append((east, north))
        elbe_km_values.append(elbekm)

# Build cumulative distances for interpolation
cumulative_distances = [0.0]
for i in range(1, len(points_proj)):
    p1 = Point(points_proj[i - 1])
    p2 = Point(points_proj[i])
    dist = p1.distance(p2)
    cumulative_distances.append(cumulative_distances[-1] + dist)

# Create LineString in projected coords
river_line = LineString(points_proj)

def get_elbe_km(lat, lon):
    """
    Given WGS84 coordinates (lat, lon), returns the interpolated elbeKM value.
    Assumes elbeKM increases downstream (as in provided dataset).
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:25833", always_xy=True)  # EPSG:25833 = UTM zone 33N (ETRS89)
    
    # Project input lat/lon to East/North
    east, north = transformer.transform(lon, lat)
    pt_proj = Point(east, north)
    
    # Project onto river line
    proj_dist = river_line.project(pt_proj)

    # Interpolate elbeKM
    for i in range(1, len(cumulative_distances)):
        if cumulative_distances[i] >= proj_dist:
            d0 = cumulative_distances[i - 1]
            d1 = cumulative_distances[i]
            if d1 == d0:
                frac = 0
            else:
                frac = (proj_dist - d0) / (d1 - d0)
            km_value = elbe_km_values[i - 1] + frac * (elbe_km_values[i] - elbe_km_values[i - 1])
            return km_value

    return elbe_km_values[-1]  # fallback for last point

