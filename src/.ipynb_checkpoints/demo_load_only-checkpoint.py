#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 09:29:54 2025

@author: fabian
"""

import geopandas as gpd
import pandas as pd
import multiprocessing
from collections import defaultdict
from shapely.geometry import Point

from geopop.tsn_problem_generator.geo_tools import *
from geopop.tsn_problem_generator.topology import Topology, Switch
import networkx as nx
import time

import json

from pyproj import Transformer

from shapely.geometry import MultiPoint

import numpy as np
import folium
import matplotlib.pyplot as plt

import geopop.loading.raster as rt
import geopop.algorithms.raster_dualtree as rd
import geopop.algorithms.hierarchy as hr

import geopop.loading.boundaries as bd
import plotly.io as pio
import geopop.plotting.plotting as pl
    
import plotly.graph_objects as go
import plotly.express as px

import webbrowser
import os

import geopandas as gpd
import pickle

def load_and_plot(config):
    
    country = config.get("country")
    

#%%read in
    output_path = config.get("folium_output")
    output_dir = os.path.dirname(os.path.abspath(output_path))


    centroids_path = os.path.join(output_dir, "df_centroids.gpkg")
    bounds_path = os.path.join(output_dir, "df_bounds.gpkg")
    
    gdf_centroids = gpd.read_file(centroids_path)
    gdf_bounds = gpd.read_file(bounds_path).rename(columns={"geometry": "bounds"})
    
    # Merge both by "id"
    df_geopop = gdf_centroids.merge(gdf_bounds[["id", "bounds"]], on="id", how="left")
    df_geopop.set_geometry("bounds", inplace=True)
    
    # At this point:
    # - gdf_combined["geometry"] = centroid
    # - gdf_combined["bounds"] = polygon
    # - All other attributes are preserved


#%%
    fig = pl.plot_hierarchy_tree(df_geopop, point_size=7, line_width=.05, colorscale='inferno')
    fig.show()
#%%
    boundary = bd.get_country_boundaries(country)
    #%%
    fig = bd.plot_with_outline_from_shape(boundary)
    fig.show()
#%%
    fig = pl.plot_polys_colored(df_geopop[df_geopop['level'] == (config.get("num_layers")-1)])
    fig.show()
    
    #%%
    pkl_path = os.path.join(output_dir, "graph.pkl")
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    df_geopop.set_geometry("geometry", inplace=True)
#%%

    def plot_graph_plotly(gdf, G, point_size=8, line_width=2, colorscale='Viridis'):
       df = gdf.copy()
   
       # Prepare base map
       center = MultiPoint(gdf.geometry).centroid
       
       epsg = gdf.crs.to_epsg()
       transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
       center_lon, center_lat = transformer.transform(center.x, center.y)
       
       fig = go.Figure()
   
       # Color setup
       num_node_layers = df["level"].max() + 1
       node_colors = px.colors.sample_colorscale(colorscale, np.linspace(0, 0.7, num_node_layers))
       edge_colors = px.colors.sample_colorscale(colorscale, np.linspace(0, 0.7, num_node_layers))#**2))
   
       # Add nodes per level
       for level in range(num_node_layers):
           nodes = df[df["level"] == level]
           color = node_colors[level]
   
           fig.add_trace(
               go.Scattermap(
                   lon=nodes["lon"],
                   lat=nodes["lat"],
                   mode='markers',
                   marker=dict(size=point_size, color=color),
                   text=[f"Node: {n}<br>Level: {level}" for n in nodes["switch"]],
                   hoverinfo='text',
                   name=f"Layer {level}"
               )
           )
   
       # Add redundant nodes (if R > 0), faded style
       redundant_nodes = df[df["R"] > 0]
       if not redundant_nodes.empty:
           fig.add_trace(
               go.Scattermap(
                   lon=redundant_nodes["lon"],
                   lat=redundant_nodes["lat"],
                   mode='markers',
                   marker=dict(size=point_size, color='red', opacity=0.4),
                   text=[f"Redundant Node: {n}" for n in redundant_nodes["switch"]],
                   hoverinfo='text',
                   name="Redundant Nodes"
               )
           )
   
       # Build lookup for coordinates
       coord_lookup = df.set_index("switch")[["lat", "lon", "level", "R"]].to_dict("index")
   
       # Edges grouped by (level1, level2)
       edge_groups = {}
       for node1, node2 in G.edges:
           if node1 not in coord_lookup or node2 not in coord_lookup:
               continue
           level_pair = tuple(sorted((coord_lookup[node1]["level"], coord_lookup[node2]["level"])))
           edge_groups.setdefault(level_pair, []).append((node1, node2))
   
       for idx, (level_pair, edges) in enumerate(edge_groups.items()):
           color = edge_colors[idx % len(edge_colors)]
           lats, lons = [], []
   
           for n1, n2 in edges:
               lat1, lon1 = coord_lookup[n1]["lat"], coord_lookup[n1]["lon"]
               lat2, lon2 = coord_lookup[n2]["lat"], coord_lookup[n2]["lon"]
               lats += [lat1, lat2, None]
               lons += [lon1, lon2, None]
   
           fig.add_trace(
               go.Scattermap(
                   lat=lats,
                   lon=lons,
                   mode="lines",
                   line=dict(color=color, width=line_width),
                   name=f"Edges {level_pair[0]}-{level_pair[1]}"
               )
           )
   
       # Mapbox layout
       fig.update_layout(
           map=dict(
               style="carto-positron",
               center={"lat": center_lat, "lon": center_lon},
               zoom=7
           ),
           margin=dict(l=0, r=0, t=0, b=0),
           showlegend=True
       )
   
       return fig
   
    fig = plot_graph_plotly(df_geopop, G)
    fig.show(renderer="notebook")
    
    
    return