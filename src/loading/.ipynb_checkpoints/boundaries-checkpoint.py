import requests
import zipfile
import io
import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import numpy as np
import rasterio 
import rasterio.plot
import pandas as pd
import cv2 as cv 
import plotly
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json 
import shapely
from shapely.geometry import shape

from pyproj import Transformer
from PIL import Image  # added for image preview

from src.config import DATAPATH

COUNTRY_TO_IS0_FILENAME = DATAPATH + 'countries.json'
#BOUNDARIES_FILENAME = DATAPATH + 'gadm_410-levels.gpkg'
from pathlib import Path
import shutil

ORIGINAL_GPKG_PATH = Path(DATAPATH) / 'gadm_410-levels.gpkg'
LOCAL_GPKG_PATH = Path('/tmp') / ORIGINAL_GPKG_PATH.name

# Copy if /tmp copy doesn't exist or is older
if not LOCAL_GPKG_PATH.exists() or ORIGINAL_GPKG_PATH.stat().st_mtime > LOCAL_GPKG_PATH.stat().st_mtime:
    print(f"[INFO] Copying {ORIGINAL_GPKG_PATH} to {LOCAL_GPKG_PATH}")
    shutil.copy2(ORIGINAL_GPKG_PATH, LOCAL_GPKG_PATH)

BOUNDARIES_FILENAME = str(LOCAL_GPKG_PATH)


with open(COUNTRY_TO_IS0_FILENAME) as f:
    countries = json.load(f)

    countries_inv = {v: k for k, v in countries.items()}

def get_countries_to_code():
    return countries

def get_country_boundaries(country):
    if country in countries.values():
        # Input is a code
        #country_name = countries[country]
        try:
            gdf = gpd.read_file(BOUNDARIES_FILENAME, layer='ADM_0')
            return gdf[gdf['GID_0'] == country]['geometry'].values[0]
        except KeyError:
            raise ValueError(f"Country '{country}' not found in the country list.")
        
    else:
        print("1", country, "1 not in country code list")
        try:
            country_name = countries_inv[country]
        except KeyError:
            if country not in countries_inv.values():
                raise ValueError(f"Country '{country}' not found in the countries list.")
            country_name = country
        
    gdf = gpd.read_file(BOUNDARIES_FILENAME, layer='ADM_0')
    return gdf[gdf['COUNTRY'] == country_name]['geometry'].values[0]

def xy_to_numpy(xy):

    xy = np.asarray(tuple(zip(*xy)))
    return xy[:, 0], xy[:, 1]

def get_shapes(geojson):
    shapes = [shape(feature['geometry']) for feature in geojson.get('features', [])]
    return shapes

def fetch_boundaries(country, level=1, preview=False):
    RELEASE_TYPE = "gbOpen"
    try:
        ISO_CODE = countries[country]
    except KeyError:
        if country not in countries.values():
            raise ValueError(f"Country '{country}' not found in the countries list.")
        ISO_CODE = country
    url = f"https://www.geoboundaries.org/api/current/{RELEASE_TYPE}/{ISO_CODE}/ADM{level}/"

    response = requests.get(url)
    if preview:
        prew_resp = requests.get(response.json()["imagePreview"])
        image = Image.open(io.BytesIO(prew_resp.content))
        fig = px.imshow(np.array(image))
        fig.show()
    if response.status_code != 200:
        raise ValueError("Failed to fetch data")
    download_link = response.json()['gjDownloadURL']
    resp2 = requests.get(download_link)

    if resp2.status_code != 200:
        raise ValueError("Failed to fetch data")
    data = resp2.json()

    return get_shapes(data)



def download_all_country_boundaries():
    """
    Downloads level 0 (country) boundaries for all countries in countries.json
    and saves them as individual JSON files in data/country_boundaries/
    """
    import os
    import json
    
    # Create the directory if it doesn't exist
    os.makedirs('data/country_boundaries', exist_ok=True)
    
    # Iterate through all countries
    for country, iso_code in countries.items():
        try:
            print(f"Fetching boundaries for {country} ({iso_code})...")
            data = fetch_boundaries(country, level=0)
            
            # Save to file
            output_path = f'data/country_boundaries/{iso_code}.json'
            with open(output_path, 'w') as f:
                json.dump(data, f)
            
            print(f"Saved {iso_code}.json")
        except Exception as e:
            print(f"Error downloading boundaries for {country} ({iso_code}): {e}")
            
    print("Download completed.")



def convert_boundaries_to_shapefiles():
    """
    Converts all GeoJSON files in data/country_boundaries/ to Shapefiles (.shp)
    and saves them in data/country_shapefiles/
    """
    import os
    import geopandas as gpd
    
    # Create output directory if it doesn't exist
    os.makedirs('data/country_shapefiles', exist_ok=True)
    
    # Get list of JSON files in the country boundaries directory
    json_dir = 'data/country_boundaries'
    if not os.path.exists(json_dir):
        print(f"Directory {json_dir} does not exist. Run download_all_country_boundaries() first.")
        return
        
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {json_dir}.")
        return
    
    for json_file in json_files:
        try:
            iso_code = json_file.replace('.json', '')
            print(f"Converting {iso_code} to Shapefile...")
            
            # Input and output file paths
            json_path = os.path.join(json_dir, json_file)
            shp_path = os.path.join('data/country_shapefiles', f"{iso_code}")
            
            # Read the GeoJSON file and save as Shapefile
            gdf = gpd.read_file(json_path)
            gdf.to_file(shp_path)
            
            print(f"Saved {iso_code}.shp")
        except Exception as e:
            print(f"Error converting {json_file} to Shapefile: {e}")
    
    print("Conversion completed.")

def get_coords(geojson):
    shapes = []
    for feature in geojson.get('features', []):
        geom = shape(feature['geometry'])
        # Handle polygons and multipolygons
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            shapes.append((x, y))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                shapes.append((x, y))
    return shapes




def plot_geojson_outline(geojson, color='blue'):
    fig = go.Figure()
    for feature in geojson.get('features', []):
        geom = shape(feature['geometry'])
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color=color, width=2),
                                     name='Polygon'))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color=color, width=2),
                                         name='MultiPolygon'))
    return fig

def plot_coords(coords, color='grey', **kwargs):
    fig = go.Figure()
    for x, y in coords:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                 line=dict(color=color, **kwargs)))
    return fig


def plot_shapes(shapes, color='blue', fig=None, **kwargs):
    if fig is None:
        fig = go.Figure()
    try:
        _ = iter(shapes)
    except TypeError:
        shapes = [shapes]
    for geom in shapes:
        if geom.geom_type == 'Polygon':
            x, y = xy_to_numpy(geom.exterior.xy)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color=color, **kwargs)))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = xy_to_numpy(poly.exterior.xy)
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color=color, **kwargs)))
    return fig

def plot_with_outline(geojson_data, mapbox_style="carto-positron",
                           center_lat=51.1657, center_lon=10.4515, zoom=3,
                           line_color="blue", line_width=2, fig=None):
    """
    Plot the outlines from GeoJSON data using Plotly's Scattermapbox.
    
    Parameters:
        geojson_data (dict): The GeoJSON data as a dictionary.
        mapbox_style (str): The mapbox style.
        center_lat (float): Latitude at the center of the map.
        center_lon (float): Longitude at the center of the map.
        zoom (int): Initial zoom level.
        line_color (str): Color of the boundary lines.
        line_width (int): Width of the boundary lines.
    """


    
    if fig is None:
        fig = go.Figure() 

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])
        
        if geom_type == "Polygon":
            # A Polygon is a list of rings; we draw each ring.
            for ring in coordinates:
                lons, lats = zip(*ring)
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=lons,
                    lat=lats,
                    line=dict(width=line_width, color=line_color),
                    name=feature.get("properties", {}).get("name", "Polygon")
                ))
        elif geom_type == "MultiPolygon":
            # A MultiPolygon is a list of polygons
            for polygon in coordinates:
                for ring in polygon:
                    lons, lats = zip(*ring)
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lon=lons,
                        lat=lats,
                        line=dict(width=line_width, color=line_color),
                        name=feature.get("properties", {}).get("name", "MultiPolygon")
                    ))
        else:
            print(f"Unsupported geometry type: {geom_type}")

    
    if fig.data and len(fig.data[0].lat) > 0 and len(fig.data[0].lon) > 0:
        center_lat = fig.data[0].lat[0]
        center_lon = fig.data[0].lon[0]

    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    fig.update_layout(showlegend=False)

    # fig.show()
    return fig

def plot_with_outline_from_shape(shapes, mapbox_style="carto-positron",
                                    center_lat=51.1657, center_lon=10.4515, zoom=10,
                                    line_color="blue", line_width=2, fig=None):
    """
    Plot the outlines from GeoJSON data using Plotly's Scattermapbox.
    
    Parameters:
        geojson_data (dict): The GeoJSON data as a dictionary.
        mapbox_style (str): The mapbox style.
        center_lat (float): Latitude at the center of the map.
        center_lon (float): Longitude at the center of the map.
        zoom (int): Initial zoom level.
        line_color (str): Color of the boundary lines.
        line_width (int): Width of the boundary lines.
    """


    
    if fig is None:
        fig = go.Figure() 

    # Polygon case
    if isinstance(shapes, shapely.geometry.Polygon):
        # Extract the coordinates of the polygon
        x, y = shapes.exterior.xy

        fig.add_trace(go.Scattermap(
                mode="lines",
                lon=np.array(x),
                lat=np.array(y),
                line=dict(width=line_width, color=line_color),
            ))

    # MultiPolygon case
    if isinstance(shapes, shapely.geometry.MultiPolygon):
        multi_x = []
        multi_y = []
        for poly in shapes.geoms:
            x, y = poly.exterior.xy
            multi_x.extend(list(x) + [None])  # Separate polygons with None
            multi_y.extend(list(y) + [None])

        fig.add_trace(go.Scattermap(
                mode="lines",
                lon=multi_x,
                lat=multi_y,
                line=dict(width=line_width, color=line_color),
            ))

        
    if fig.data and len(fig.data[0].lat) > 0 and len(fig.data[0].lon) > 0:
        center_lat = fig.data[0].lat[0]
        center_lon = fig.data[0].lon[0]
        print(center_lat, center_lon)

    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    fig.update_layout(showlegend=False)

    # fig.show()
    return fig


def plot_tile(img, fig=None, lower_left=None, upper_right=None):
    # Define a discrete colorscale with step changes.
    # This maps:
    # 0 ≤ value < 1 to 'black'
    # 1 ≤ value < 2 to 'brown'
    # 2 ≤ value < 3 to 'blue'
    # 3 ≤ value ≤ 256 to 'grey'
    if fig is None:
        fig = go.Figure()
    colorscale = [
        [0.0, 'black'],
        [1/256, 'black'],
        [1/256, 'brown'],
        [2/256, 'brown'],
        [2/256, 'blue'],
        [3/256, 'blue'],
        [3/256, 'grey'],
        [1.0, 'grey']
    ]
    heatmap_kwargs = {
        "z": img,
        "colorscale": colorscale,
        "zmin": 0,
        "zmax": 256,
        "colorbar": dict(
            tickmode='array',
            tickvals=[0, 1, 2, 255],
            ticktext=['0', '1', '2', '255']
        )
    }
    if lower_left is not None and upper_right is not None:
        # lower_left and upper_right are expected as (longitude, latitude)
        n_rows, n_cols = img.shape
        x_vals = np.linspace(lower_left[0], upper_right[0], n_cols)
        y_vals = np.linspace(lower_left[1], upper_right[1], n_rows)
        heatmap_kwargs["x"] = x_vals
        heatmap_kwargs["y"] = y_vals
    
    fig.add_traces(data=go.Heatmap(**heatmap_kwargs))
    
    return fig

def convert_3035_to_4326(x, y):
    # EPSG:3035 (ETRS89 / LAEA Europe) to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat
    
def extract_lower_left(text: str):
    import re
    # Pattern breakdown:
    # - "Lower Left" literal.
    # - First parenthesized group: two numbers for the numeric coordinates.
    # - Second parenthesized group: two parts separated by a comma for the DMS coordinates.
    pattern = r"Lower Left\s*\(\s*([\d\.]+),\s*([\d\.]+)\)\s*\(\s*([^,]+),\s*([^)]+)\)"
    match = re.search(pattern, text)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        # lon_dms = match.group(3).strip()
        # lat_dms = match.group(4).strip()
        return convert_3035_to_4326(x, y)
    
    raise ValueError("No match found")

def download_and_unzip(url, img_shape=(500, 500)):

    if not os.path.exists('tiles'):
        os.mkdir('tiles')

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download file: {response.status_code}")
        return np.zeros(img_shape), (0, 0)

    with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
        for file in thezip.namelist():
            if file.lower().endswith('.tif'):
                thezip.extract(file, path='tiles')
                tiff = rasterio.open(os.path.join('tiles', file))
                img = cv.resize(tiff.read(1), img_shape)
            elif 'metadata' in file.lower():
                thezip.extract(file, path='tiles')
                with open(os.path.join('tiles', file), 'r') as f:
                    lon, lat = extract_lower_left(f.read())

        return img, (lon, lat)

def fetch_tile(tile_id, resolution_meter=10, **kwargs):
    match resolution_meter:

        case 2:
            url = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/ESM_BUILT_VHR2015_Europe_R2019/ESM_BUILT_VHR2015_EUROPE_R2019_3035_02/V1-0/tiles/{tile_id}.zip"
        case 10:
            url = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/ESM_BUILT_VHR2015_Europe_R2019/ESM_BUILT_VHR2015CLASS_EUROPE_R2019_3035_10/V1-0/tiles/{tile_id}.zip"

        case _:
            print(f"Resolution {resolution_meter} not supported")
   
    return download_and_unzip(url, **kwargs)




def match_shapes(shapes1, shapes2):
    matches = {i : [] for i in range(len(shapes1))}

    shapes1_buffered = [shape.buffer(1e-8, join_style=2) for shape in shapes1]
    for j, shape2 in enumerate(shapes2):
        for i, shape1 in enumerate(shapes1_buffered):
            if shape2.within(shape1):
            # if shape2.intersects(shape1):
                matches[i].append(j)
                break
    return matches

