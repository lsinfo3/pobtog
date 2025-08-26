#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:00:51 2025

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
import pickle

#%%
# Link planning functions
def plan_ring_fast(node_list):
    links = []
    if len(node_list) < 2:
        return links
    if len(node_list)==2:
        links.append((node_list[0], node_list[1]))
        return links

    start_node = min(node_list, key=lambda node: node.location.x)
    unvisited = set(node_list)
    unvisited.remove(start_node)
    ring_order = [start_node]
    current_node = start_node

    while unvisited:
        next_node = min(unvisited, key=lambda node: current_node.location.distance(node.location))
        unvisited.remove(next_node)
        ring_order.append(next_node)
        current_node = next_node

    for i in range(len(ring_order)):
        node1 = ring_order[i]
        node2 = ring_order[(i + 1) % len(ring_order)]
        links.append((node1, node2))

    return links


def plan_ring_tsp_networkx(node_list):
    if len(node_list) < 2:
        return []
    if len(node_list) == 2:
        return [(node_list[0], node_list[1])]

    G = nx.Graph()
    for i, node1 in enumerate(node_list):
        for j in range(i + 1, len(node_list)):
            node2 = node_list[j]
            dist = node1.location.distance(node2.location)
            G.add_edge(node1, node2, weight=dist)

    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=True, method= nx.approximation.christofides)
    links = [(tsp_path[i], tsp_path[(i + 1) % len(tsp_path)]) for i in range(len(tsp_path))]
    return links


def plan_star(center_node, node_list):
    return [(center_node, node) for node in node_list]

def plan_fully_connected(node_list):
    links = []
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if i < j:
                links.append((node1, node2))
    return links

def plan_connections(connection_type, nodes):
    if connection_type == "ring":
        return plan_ring_tsp_networkx(nodes)
    elif connection_type == "star":
        center = nodes[0]
        others = nodes[1:]
        return plan_star(center, others)
    elif connection_type == "full":
        return plan_fully_connected(nodes)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

def worker(args):
    switches, connection_type = args
    return plan_connections(connection_type, switches)
#%%

def run_calculation_pipeline(config):

    #%%
    country = config.get("country")
    raster, transform = rt.load_country_raster_splitted_mp(country)
    
    #%%

    df_dualtree = rd.dualtree_on_rasters(raster, transform, capacity=config.get("DT_max_capacity"))
    print('Number of remaining pixels: ', df_dualtree.shape[0])
    df_dualtree.head(3)
    
    #%%

    boundary = bd.get_country_boundaries(country)
    #%%

    
    def generate_cluster_sizes(N=10000, L=4, core=5, alpha=4.5):
        if L < 2:
            raise ValueError("L must be at least 2.")
        if N <= core:
            raise ValueError("Top size N must be larger than core.")
        
        # Generate sizes using power decay
        sizes = [
            round(core + ((L - 1 - i) / (L - 1)) ** alpha * (N - core))
            for i in range(L)
        ]
        sizes[-1] = core  # ensure exact match at bottom layer
        
        # Assert strictly decreasing sizes
        if not all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1)):
            raise ValueError(
                f"Invalid cluster sizes: {sizes}. "
                "Layer sizes must be strictly decreasing. "
                "Please manually specify appropriate layer sizes."
            )
        
        return sizes[::-1]
    
    
    
    
    n_clusters = config.get("num_lower_layer_nodes")
    if n_clusters ==[]: n_clusters = generate_cluster_sizes(df_dualtree.shape[0], config.get("num_layers"))
    
    print("Cluster sizes", n_clusters)
    
    
    
    
    
    #cluster1 = df_dualtree.shape[0] // 30
    #cluster2 = cluster1 *50 // 400
    #cluster3 = cluster2 // 5
    #print(cluster1, cluster2, cluster3)
    df_geopop = hr.kmeans_tree_from_points(df_dualtree.values,boundary, n_clusters=n_clusters[::-1])
    #df_geopop = hr.kmeans_tree_from_points(df_dualtree.values, boundary, n_clusters=[cluster1, cluster2, cluster3])
    print(df_geopop.shape)
    df_geopop.head(3)
    
    #%%

    #pio.renderers.default="notebook"


    fig = pl.plot_hierarchy_tree(df_geopop, point_size=7, line_width=.05, colorscale='inferno')
    fig.show()
    
    #%%

    boundary = bd.get_country_boundaries(country)
    #%%
    fig = bd.plot_with_outline_from_shape(boundary)
    fig.show()
    #%%
    hr.compute_voronoi(df_geopop, boundary)
    df_geopop.head(3)
    
    #%%
    fig = pl.plot_polys_colored(df_geopop[df_geopop['level'] == (config.get("num_layers")-1)])
    fig.show()
    
    #%%
    
    begin_time = time.time()
    start_time = time.time()
    print(f"Starting preprocessing topology")
    
    
    df= df_geopop.copy()
    
    #%%
    
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    df = df.drop(columns = ["lon","lat"])
    
    
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326") 
    gdf = gdf.to_crs(epsg=3857)
    gdf["R"] = 0
    
    #%%
    #%%capacity level
    
    num_layers = config.get("num_layers")
    level_to_capacity = dict(enumerate(config["link_capacity"]))
    gdf["capacity"] = gdf["level"].map(level_to_capacity)
    
    #%%
    
    #copy the nodes for redundancy and shift their position slightly
    #Todo: put this into a function called add_redundant_nodes
    
    for i in range(0,num_layers):
        redundancy = config.get("redundancy")[i]
        copies_list = []
        for r in range(1,redundancy):
            copies = gdf[gdf["level"]==i].copy()
            copies["R"] = r
            copies["geometry"] = copies["geometry"].apply(lambda geom: Point(geom.x + r, geom.y))
            copies_list.append(copies)
        
        if copies_list !=[]:
            gdf_copies = pd.concat(copies_list, ignore_index=True)
            gdf = pd.concat([gdf, gdf_copies], ignore_index=True)
    
    #%%give every node a clear topology switch
    gdf["switch"] = gdf.apply(lambda x: Switch(f"{x.id}_R{x.R}", x.geometry),axis=1)
    
    switch_to_capacity = dict(zip(gdf["switch"], gdf["capacity"]))
    
    
    
    #%% #%% iteratively connect layers, starting with the core. then for each node find the successors on the next layer and connect them according to the config.
    #ToDo: put this into a function which takes the gdf and the config
    
    elapsed = time.time() - start_time
    print(f"Finished preprocessing in {elapsed:.2f} seconds")
    G = nx.Graph()
    
    save_to_json = config.get("save_to_json")
    if save_to_json:
        #json_nodes = [
        #{
        #    "name": node.name,
        #    "type": node.type,
        #    "location": [node.location.x, node.location.y]
        #}
        #for node in gdf["switch"]
        #]
        json_nodes = [
        {
            "name": row.switch.name,
            "type": row.switch.type,
            "location": [row.switch.location.x, row.switch.location.y],
            "level": row.level,     # <-- kommt aus gdf["level"]
        }
        for row in gdf.itertuples(index=False)
        ]
        json_links = []
    
    
    total_layer_time = 0
    total_link_add_time = 0
    
    #add all nodes
    G.add_nodes_from([(node, {"origin": node}) for node in gdf["switch"]])
    
    
    for i in range(0, num_layers):
        layer_start_time = time.perf_counter()
        start_time = time.time()
        print(f"Processing layer {i}")
        
        
        connection = config.get("connection")[i]
        link_capacity = config.get("link_capacity")[i]
        current_layer = gdf[gdf["level"] == i]
        redundancy_level = config["redundancy"][i]
        if i == 0: #core
    
            for redundancy_group_id in range(0,redundancy_level):
                red_group = current_layer[current_layer["R"] == redundancy_group_id]
                red_group = red_group["switch"].tolist()
                # Plan and immediately apply (no multiprocessing needed)
                links = plan_connections(connection, red_group)
                for node1, node2 in links:
                    bandwidth = min(switch_to_capacity[node1], switch_to_capacity[node2])
                    
                    start_link_time = time.perf_counter()
    
                    if save_to_json:
                        json_links.append({"n1": node1.name, "n2": node2.name, "bandwidth": bandwidth})
                        json_links.append({"n1": node2.name, "n2": node1.name, "bandwidth": bandwidth})
                    G.add_edge(node1, node2, bandwidth=bandwidth)
                    
                    
                    
    
                    link_time = time.perf_counter() - start_link_time
                    total_link_add_time += link_time
                
                    
        else:  # Non-core layers
            prev_redundancy_level = config["redundancy"][i - 1]
            parents_layer = gdf[gdf["level"] == i - 1]
            tasks = []  # Prepare all tasks first, then do one multiprocessing run
    
            for pre_redundancy_group_id in range(prev_redundancy_level):
                parents = parents_layer[parents_layer["R"] == pre_redundancy_group_id]
    
                for redundancy_group_id in range(redundancy_level):
                    red_group = current_layer[current_layer["R"] == redundancy_group_id]
                    # Build parent → child mappings
                    parent_child_mapping = defaultdict(list)
                    for node in red_group.itertuples(index=False):
                        parent_child_mapping[node.parent_id].append(node.switch)
    
    
                    for parent_id, children_switches in parent_child_mapping.items():
                        parent_row = parents[parents["id"] == parent_id]
                        if not parent_row.empty:
                            parent_switch = parent_row.iloc[0]["switch"]
                            switches = [parent_switch] + children_switches
                            tasks.append((switches, connection))
    
            # Run all link plannings in parallel ONCE per layer
            if tasks:
                with multiprocessing.Pool() as pool:
                    results = pool.map(worker, tasks)
    
                # Sequentially add links
                for link_list in results:
                    for node1, node2 in link_list:
                        bandwidth = min(switch_to_capacity[node1], switch_to_capacity[node2])
    
                        
                        start_link_time = time.perf_counter()
    
                        if save_to_json:
                            json_links.append({"n1": node1.name, "n2": node2.name, "bandwidth": bandwidth})
                            json_links.append({"n1": node2.name, "n2": node1.name, "bandwidth": bandwidth})
                        G.add_edge(node1, node2, bandwidth=bandwidth)
    
                        link_time = time.perf_counter() - start_link_time
                        total_link_add_time += link_time
        layer_elapsed = time.perf_counter() - layer_start_time
        total_layer_time += layer_elapsed
        elapsed = time.time() - start_time
        print(f"Finished layer {i} in {elapsed:.2f} seconds")
        
    if save_to_json:
        topo_json = {
        "nodes": json_nodes,
        "links": json_links,
        "streams": []
        }
        
        # 4. Write to file
        output_file = config.get("json_output")
        with open(output_file, "w") as f:
            json.dump(topo_json, f, indent=4)
    
    print(f"Total layer processing time: {total_layer_time:.2f}s")
    print(f"Total time spent in 'topo.create_and_add_links': {total_link_add_time:.2f}s")
    print(f"Fraction of time in link creation: {total_link_add_time / total_layer_time:.2%}")
    
    #print(len(G.nodes))
    #print(len(json_nodes))
    #print(len(G.edges))
    #print(len(json_links))
    
    
    #%%
    
    def generate_folium_map(output_path, G, gdf):
        epsg = gdf.crs.to_epsg()
        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    
        # Vectorized transform of all geometries
        coords = np.array([transformer.transform(pt.x, pt.y) for pt in gdf.geometry])
        gdf["lon"] = coords[:, 0]
        gdf["lat"] = coords[:, 1]
    
        # Precompute map center
        centroid = MultiPoint(gdf["geometry"]).centroid
        centroid_lon, centroid_lat = transformer.transform(centroid.x, centroid.y)
        m = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=10, tiles="cartodbpositron")
    
        # Extract values as arrays
        levels = gdf["level"].values
        R_flags = gdf["R"].values > 0
        switches = gdf["switch"].values
        latitudes = gdf["lat"].values
        longitudes = gdf["lon"].values
    
        # Prepare node layers
        num_node_layers = levels.max() + 1
        node_colors = plt.cm.viridis(np.linspace(0, 0.7, num_node_layers))
        node_layers = [folium.FeatureGroup(name=f"Layer_{i}") for i in range(num_node_layers)]
        redundant_node_layer = folium.FeatureGroup(name="Redundant Nodes", show = False)
    
        # Build lookup: switch -> (lat, lon, level, R)
        node_lookup = {}
        for lat, lon, level, is_red, node in zip(latitudes, longitudes, levels, R_flags, switches):
            node_lookup[node] = (lat, lon, level, is_red)
            node_color = f"#{int(node_colors[level][0]*255):02x}{int(node_colors[level][1]*255):02x}{int(node_colors[level][2]*255):02x}"
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.7,
                popup=f"Node: {node.name}, Level: {level}, Redundancy: {int(is_red)}"
            )
            if is_red:
                redundant_node_layer.add_child(marker)
            else:
                node_layers[level].add_child(marker)
    
        # Prepare edge layers
        edge_layers = {}
        edge_keys = []
        for i in range(num_node_layers):
            for j in range(i, num_node_layers):
                key = (i, j)
                edge_keys.append(key)
                edge_layers[key] = folium.FeatureGroup(name=f"Edges_Layer_{i}-{j}")
        redundant_edge_layer = folium.FeatureGroup(name="Redundant Edges", show = False)
    
        edge_colors = plt.cm.viridis(np.linspace(0, 0.7, len(edge_keys)))
    
        # Add edges
        for node1, node2 in G.edges:
            if node1 not in node_lookup or node2 not in node_lookup:
                continue
    
            lat1, lon1, level1, red1 = node_lookup[node1]
            lat2, lon2, level2, red2 = node_lookup[node2]
            layer_key = tuple(sorted((level1, level2)))
            color_idx = edge_keys.index(layer_key)
            edge_color = edge_colors[color_idx]
            edge_color_hex = f"#{int(edge_color[0]*255):02x}{int(edge_color[1]*255):02x}{int(edge_color[2]*255):02x}"
    
            polyline = folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color=edge_color_hex,
                weight=2,
                opacity=0.7
            )
    
            if red1 or red2:
                redundant_edge_layer.add_child(polyline)
            else:
                edge_layers[layer_key].add_child(polyline)
    
        # Reverse layer order: lowest level rendered last = on top
        for layer in reversed(node_layers):
            if len(layer._children) > 0:
                layer.add_to(m)
    
        for layer in reversed(list(edge_layers.values())):
            if len(layer._children) > 0:
                layer.add_to(m)
    
        if len(redundant_node_layer._children) > 0:
            redundant_node_layer.add_to(m)
    
        if len(redundant_edge_layer._children) > 0:
            redundant_edge_layer.add_to(m)
    
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(output_path)
        print(f"Map saved at: {output_path}")
    
    generate_folium_map(config.get("folium_output"), G, gdf)

    if True:
    #if(config.get("folium_show") == True):
        output_path = config.get("folium_output")
        # Open the saved HTML file in a new browser tab
        file_url = f"file://{os.path.abspath(output_path)}"
        webbrowser.open_new_tab(file_url)

    
    elapsed = time.time() - begin_time
    print(f"Finished topology in {elapsed:.2f} seconds")
    
    
    #%%

    
    def add_lat_lon_columns(gdf):
        """Adds 'lat' and 'lon' columns to GeoDataFrame using coordinate transform."""
        epsg = gdf.crs.to_epsg()
        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    
        coords = np.array([transformer.transform(pt.x, pt.y) for pt in gdf.geometry])
        gdf["lon"] = coords[:, 0]
        gdf["lat"] = coords[:, 1]
        return gdf
    
    gdf = add_lat_lon_columns(gdf)
    
    
    
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
    
        
    
    fig = plot_graph_plotly(gdf, G)
    fig.show(renderer="notebook")
    
    output_path = config.get("folium_output")
    output_dir = os.path.dirname(os.path.abspath(output_path))


    centroids_path = os.path.join(output_dir, "df_centroids.gpkg")
    bounds_path = os.path.join(output_dir, "df_bounds.gpkg")
    
    if os.path.exists(bounds_path):
        os.remove(bounds_path)
        
    if os.path.exists(centroids_path):
        os.remove(centroids_path)

    # Save centroids (drop "bounds")
    gdf_centroids = gdf.drop(columns=["bounds"]).set_geometry("geometry")
    gdf_centroids.to_file(centroids_path, driver="GPKG")

    # Save bounds (drop "geometry")
    gdf_bounds = gdf.drop(columns=["geometry"]).set_geometry("bounds")
    gdf_bounds.to_file(bounds_path, driver="GPKG")
        
    #save df
    #save G
    pkl_path = os.path.join(output_dir, "graph.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    
    return