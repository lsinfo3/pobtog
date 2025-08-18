import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import shapely
from tqdm import tqdm, trange
from random import shuffle, randint
from pyproj import Geod
from .raster_dualtree import RasterDualTree
from sklearn.cluster import KMeans, BisectingKMeans 
import mapclassify


from src.utils.geoutils import idx_to_lonlat
from src.loading.raster import load_country_raster, filter_polys

from .dualtree_hierarchy import * 
import dataclasses
from collections import deque


def flip_levels(df):

    max_level = df['level'].max()
    df['level'] = max_level - df['level']
    return df

def assign_ancestors(df, max_depth):

    # Initialize ancestor columns with NaN
    for i in range(max_depth):
        df[f'ancestor{i}'] = np.nan
    
    # Create a mapping of id to index for quick lookups
    id_to_idx = {id_val: idx for idx, id_val in enumerate(df['id'])}
    
    # For each row in the dataframe
    for idx, row in df.iterrows():
        # Start with the direct parent
        parent_id = row['parent_id']
        
        # Trace back through all ancestors
        while not pd.isna(parent_id):
            # Find the parent's index and get its level
            if parent_id in id_to_idx:
                parent_idx = id_to_idx[parent_id]
                parent_level = int(df.at[parent_idx, 'level'])
                
                # Record this ancestor at its proper level
                if 0 <= parent_level < max_depth:  # Ensure level is within valid range
                    df.at[idx, f'ancestor{parent_level}'] = parent_id
                
                # Move up to parent's parent
                parent_id = df.at[parent_idx, 'parent_id']
            else:
                parent_id = np.nan
    


def kmeans_tree_from_raster(raster, transform, capacity=10_000, n_clusters=None):
    centers = RasterDualTree.build_from_raster(raster, capacity=capacity, return_tree=False)
    idx_to_lonlat(centers, inplace=True, transform=transform)

    return kmeans_tree_from_points(centers, n_clusters=n_clusters)

def crop_to_shape(df, shape, simplify_tol=None):
    try:
        polys = [poly for poly in shape.geoms]
    except AttributeError:
        polys = [shape]
    polys = filter_polys(polys, simplify_tol=simplify_tol)
    boundary = shapely.union_all(polys)
    # for i, row in df.iterrows():
    #     df.loc[i, 'bounds'] = shapely.intersection(row.bounds, boundary)
    # df['bounds'] = gpd.GeoSeries(df['bounds']).intersection(boundary)
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs='EPSG:4326')
    df['bounds'] = gpd.overlay(df['bounds'], boundary_gdf, how='intersection')


#def compute_voronoi(df):
#    # Create bounds for each level using Voronoi polygons - vectorized approach
#    bounds = pd.Series(np.empty(len(df)), index=df.index)
#    for level in df['level'].unique():
#        level_df = df[df['level'] == level]
#        level_indices = level_df.index
#        
#        if len(level_df) > 1:  # Need at least 2 points for Voronoi
#            try:
#                # Extract coordinates as arrays
#                coords = level_df[['lon', 'lat']].values
#                
#                # Create Voronoi diagram
#                multi_point = shapely.geometry.MultiPoint(coords)
#                voronoi_regions = shapely.voronoi_polygons(multi_point)
#                
#                # Assign all polygons at once to the dataframe
#                # df.loc[level_indices, 'bounds'] = voronoi_regions.geoms
#                bounds[level_indices] = voronoi_regions.geoms
#                
#            except Exception as e:
#                print(f"Error creating Voronoi for level {level}: {e}")
#                # Create points and buffers in a vectorized way
#                points = shapely.points(level_df['lon'].values, level_df['lat'].values)
#                buffers = shapely.buffer(points, 0.01)
#                # df.loc[level_indices, 'bounds'] = buffers
#                bounds[level_indices] = buffers
#        else:
#            # For single point, create a buffer
#            point = shapely.points(level_df['lon'].values, level_df['lat'].values)[0]
#            # df.loc[level_indices[0], 'bounds'] = shapely.buffer(point, 0.01)
#            bounds[level_indices[0]] = shapely.buffer(point, 0.01)


def compute_voronoi(df, boundary):
    # Create bounds for each level using Voronoi polygons - vectorized approach
    for level in df['level'].unique():
        level_df = df[df['level'] == level]
        level_indices = level_df.index
        
        if len(level_df) > 1:  # Need at least 2 points for Voronoi
            try:
                # Extract coordinates as arrays
                coords = level_df[['lon', 'lat']].values
                
                # Create Voronoi diagram
                multi_point = shapely.geometry.MultiPoint(coords)
                voronoi_regions = shapely.voronoi_polygons(multi_point)

                country_union = shapely.ops.unary_union(boundary).simplify(0.01, preserve_topology=True)
                intersected_regions = [region.intersection(country_union) for region in voronoi_regions.geoms]

                # Assign all polygons at once to the dataframe
                df.loc[level_indices, 'bounds'] = intersected_regions#list(voronoi_regions.geoms)
                
            except Exception as e:
                print(f"Error creating Voronoi for level {level}: {e}")
                # Create points and buffers in a vectorized way
                points = shapely.points(level_df['lon'].values, level_df['lat'].values)
                buffers = shapely.buffer(points, 0.01)
                df.loc[level_indices, 'bounds'] = buffers
        else:
            # For single point, create a buffer
            point = shapely.points(level_df['lon'].values, level_df['lat'].values)[0]
            df.loc[level_indices[0], 'bounds'] = shapely.buffer(point, 0.01)
    

def kmeans_tree_from_points(points, boundary, n_clusters=None):
    
    # Initial dataframe with leaf points
    df = pd.DataFrame(points, columns=['lon', 'lat', 'pop'])
    df['level'] = 0
    df['parent_id'] = np.nan
    df['id'] = np.arange(1, len(df) + 1)  # Assign unique IDs
    df['is_leaf'] = True
    df['bounds'] = None  # Initialize bounds column
    
    centroids_pop = points[:, 2]
    centers = points[:, :2]
    
    if n_clusters is None:
        n_clusters = list(reversed([4] + [16**i for i in range(1, 4)]))
    id_counter = len(df)

    if len(centers) < n_clusters[0]:
        raise ValueError(f"Number of points ({len(centers)}) is less than the number of clusters ({n_clusters[0]}).")
    
    # Build the hierarchy
    for i in trange(len(n_clusters)):
        kmeans = BisectingKMeans(n_clusters=n_clusters[i], bisecting_strategy='largest_cluster')
        kmeans.fit(centers, sample_weight=centroids_pop)
        
        # Create new parent nodes
        centroids_pop = np.bincount(kmeans.labels_, weights=centroids_pop)
        centers = kmeans.cluster_centers_

        aligned_centers = align_nodes_on_land(centers, boundary)
        
        df_new = pd.DataFrame(aligned_centers, columns=['lon', 'lat'])
        df_new['pop'] = centroids_pop
        df_new['level'] = i + 1
        df_new['parent_id'] = np.nan
        df_new['id'] = np.arange(id_counter + 1, id_counter + len(df_new) + 1)
        df_new['is_leaf'] = False
        df_new['bounds'] = None
        id_counter += len(df_new)
        
        # Update parent IDs for current level nodes
        lower_level_mask = df['level'] == i
        df.loc[lower_level_mask, 'parent_id'] = [df_new.iloc[label]['id'] for label in kmeans.labels_]
        
        df = pd.concat([df, df_new], ignore_index=True)
    
    # Flip levels before computing ancestors
    flip_levels(df)
    
    # Now compute ancestor columns
    max_depth = len(n_clusters)
    
    assign_ancestors(df, max_depth)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='bounds', crs='EPSG:4326')
    
    return gdf

#def align_nodes_on_land(numpy_coords, boundary):
#    i, j = 0,0 
#    land_area = shapely.ops.unary_union(boundary).simplify(0.01, preserve_topology=True) #boundary.unary_union
#    corrected_coords = []
#    for lon, lat in numpy_coords:
#        point = shapely.Point(lon, lat)
#        
#        # If point is in land, keep it
#        if land_area.contains(point):
#            i += 1
#            corrected_coords.append((lon, lat))
#        else:
#            # Otherwise find the nearest point on land
#            j += 1
#            nearest_geom = shapely.ops.nearest_points(point, land_area)[1]
#            corrected_coords.append((nearest_geom.x, nearest_geom.y))
#    print(f'{i}/{i + j} nodes were directly added and {j}/{i + j} had to be realligned')
#    return np.array(corrected_coords)

def align_nodes_on_land(numpy_coords, boundary):
    land_area = shapely.ops.unary_union(boundary)#.simplify(0.01, preserve_topology=True)
    
    # Extract lon and lat as separate arrays
    lons = numpy_coords[:, 0]
    lats = numpy_coords[:, 1]
    
    # Vectorized containment check
    mask = shapely.contains_xy(land_area, lons, lats)
    
    # Directly use coordinates that are inside land
    corrected_coords = numpy_coords.copy()
    
    # Points outside the land area
    off_land_points = numpy_coords[~mask]
    
    # Apply nearest_points to each off-land point (still a loop here)
    for idx, (lon, lat) in zip(np.where(~mask)[0], off_land_points):
        point = shapely.Point(lon, lat)
        nearest_geom = shapely.ops.nearest_points(point, land_area)[1]
        corrected_coords[idx] = (nearest_geom.x, nearest_geom.y)

    i = np.sum(mask)
    j = np.sum(~mask)
    print(f'{i}/{i + j} nodes were directly added and {j}/{i + j} had to be realligned')

    return corrected_coords


if __name__ == '__main__':

    country = 'Indonesia'
    # raster, transform = load_country_raster_mainland(country)
    raster, transform = load_country_raster_splitted(country)
    # _, tree = RasterDualTree.build_from_raster(raster, capacity=10_000, return_tree=True, transform=transform)
    # gdf = dualtree_to_hierarchy(tree)
    # fig = plot_regions(gdf, level=-1, region_level=5)
    # fig.show()

    dfk = kmeans_tree_from_raster(raster, transform, capacity=100_000, n_clusters=[50, 4])
    crop_to_country_boundary(dfk, country)
    # fig = plot_regions(dfk, level=-1, region_level=1)
    fig = plot_polygon_bounds(dfk[dfk['level'] == 1])
    fig.write_html('plot.html')
    fig.show()