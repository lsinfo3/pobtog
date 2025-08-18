import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import rasterio
from pyproj import Geod
from shapely.geometry import Polygon, LineString, MultiPolygon

def bbox_area(bbox):
    
    minx, miny, maxx, maxy = bbox
    
    # Initialize Geod with WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    
    # Create polygon coordinates (lon, lat) for the bounding box
    lons = [minx, maxx, maxx, minx, minx]  # Close the polygon
    lats = [miny, miny, maxy, maxy, miny]  # Close the polygon
    
    # Calculate the geodesic area
    area, _ = geod.polygon_area_perimeter(lons, lats)
    
    # Convert to square kilometers
    area_sq_km = abs(area) / 1_000_000  # abs() because area can be negative based on orientation
    
    return area_sq_km

def bboxes_area(bboxes):
    total_area = 0
    for bbox in bboxes:
        area = bbox_area(bbox)
        total_area += area
    return total_area


def get_bbox_from_raster(raster, transform):
    height, width = raster.shape if len(raster.shape) == 2 else raster.shape[1:3]
    
    # Get coordinates of the four corners
    # Upper left corner (pixel coordinates 0, 0)
    ul_x, ul_y = transform * (0, 0)
    
    # Lower right corner (pixel coordinates width, height)
    lr_x, lr_y = transform * (width, height)
    
    # Create bbox (minx, miny, maxx, maxy)
    # Note: depending on the orientation of your transform, 
    # you might need to adjust min/max calculation
    minx = min(ul_x, lr_x)
    miny = min(ul_y, lr_y)
    maxx = max(ul_x, lr_x)
    maxy = max(ul_y, lr_y)
    
    return (minx, miny, maxx, maxy)

def raster_area(raster, transform):
    bbox = get_bbox_from_raster(raster, transform)
    area = bbox_area(bbox)
    
    return area

def rasters_area(rasters, transforms):
    total_area = 0
    for raster, transform in zip(rasters, transforms):
        area = raster_area(raster, transform)
        total_area += area
    return total_area
    
def idx_to_lonlat_tuple(coords, transform):
    res = idx_to_lonlat(np.array(coords).reshape(-1, 2), transform, inplace=False)
    return res[0, 0], res[0, 1]

def idx_to_lonlat(coords, transform, inplace=True):
    xs, ys = rasterio.transform.xy(transform, coords[:, 0], coords[:, 1])
    if not inplace:
        coords = coords.copy()

    coords[:, 0] = xs
    coords[:, 1] = ys

    return coords


def read_pop_density(file, **kwargs):

    df = pd.read_csv(file, **kwargs)
    df.columns = [x.lower() for x in df.columns]
    df.rename(columns={'longitude' : 'lon', 'latitude' : 'lat', df.columns[-1] : 'pop'}, inplace=True)
    df.drop(df.columns[2:-1], axis=1, inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(np.float32)
    

    
    return df

def downsample_shapes(shapes, num_points):
    """
    Downsamples a list of Shapely shapes so that each shape
    has at most `num_points` vertices.
    
    For Polygons, the exterior ring is downsampled and the polygon is recreated.
    For MultiPolygons, each polygon's exterior ring is downsampled and the MultiPolygon is recreated.
    For LineStrings, the line is downsampled.
    Other geometry types are returned unchanged.
    
    Parameters:
        shapes (list): List of Shapely geometry objects.
        num_points (int): Desired number of vertices (including closing point for polygons).
        
    Returns:
        list: List of downsampled Shapely geometry objects.
    """
    new_shapes = []
    for shape in shapes:
        if shape.geom_type == 'Polygon':
            coords = list(shape.exterior.coords)
            if len(coords) <= num_points:
                new_shapes.append(shape)
                continue
            indices = np.linspace(0, len(coords) - 1, num=num_points, dtype=int)
            new_coords = [coords[i] for i in indices]
            if new_coords[0] != new_coords[-1]:
                new_coords.append(new_coords[0])
            new_shapes.append(Polygon(new_coords))
        elif shape.geom_type == 'MultiPolygon':
            new_polygons = []
            for poly in shape.geoms:
                coords = list(poly.exterior.coords)
                if len(coords) <= num_points:
                    new_polygons.append(poly)
                else:
                    indices = np.linspace(0, len(coords) - 1, num=num_points, dtype=int)
                    new_coords = [coords[i] for i in indices]
                    if new_coords[0] != new_coords[-1]:
                        new_coords.append(new_coords[0])
                    new_polygons.append(Polygon(new_coords))
            new_shapes.append(MultiPolygon(new_polygons))
        elif shape.geom_type == 'LineString':
            coords = list(shape.coords)
            if len(coords) <= num_points:
                new_shapes.append(shape)
                continue
            indices = np.linspace(0, len(coords) - 1, num=num_points, dtype=int)
            new_coords = [coords[i] for i in indices]
            new_shapes.append(LineString(new_coords))
        else:
            new_shapes.append(shape)
    return new_shapes
