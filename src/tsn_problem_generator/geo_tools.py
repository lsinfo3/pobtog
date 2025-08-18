#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:23:50 2025

@author: fabian
"""

from geopop.tsn_problem_generator.topology import Topology, Switch, Host
from shapely.geometry import Point
import folium
from pyproj import Transformer
import matplotlib.pyplot as plt
import numpy as np
import random


#connect deeper layer with higher layer
def connect_layers_with_config(topo,layer_low, layer_high, layer_config):
    area_ident = layer_config.get("area_ident")
    df_area = layer_config.get("df_area")
    connection_type = layer_config.get("intra_layer")
    bandwidth = layer_config.get("capacities")
    
    area_keys = []
    layer_low_map = {} #map of layer_low nodes, a list for each area_key
    layer_high_map = {} #map of layer_high nodes, a list for each area_key
    node_mappings={} #map with final layer_low to layer_high mappings
    used_high_nodes = set()
    if area_ident ==None:
        if len(layer_low)>len(layer_high):
            layer_low = random.sample(layer_low, len(layer_high))  # Reduce to match size 
        area_keys = ["global"] #then we have only one area identifier and its global
        layer_low_map["global"] = layer_low
        layer_high_map["global"] = layer_high
        
    else:
        area_keys = df_area[area_ident].unique()
   
        for area_key in area_keys:
            shape = df_area[df_area[area_ident] == area_key]["geometry"].values[0]
            layer_low_map[area_key] = [node for node in layer_low if node.location.within(shape)]    #map each layer_low node to an area_key based on which geometry they are in
            layer_high_map[area_key] = [node for node in layer_high if node.location.within(shape)]  #map each layer_low node to an area_key based on which geometry they are in

    #within each area key assign layer_high nodes to layer_low nodes and keep a mapping (list of layer_high nodes) for each layer_low node
    for area_key in area_keys: 
        
        low_nodes = layer_low_map[area_key]
        high_nodes = layer_high_map[area_key]
        node_mappings[area_key] = {}
        if not low_nodes or not high_nodes:
            continue  # Skip if no valid nodes in this area
        if len(low_nodes) > len(high_nodes):
            low_nodes = random.sample(low_nodes, len(high_nodes)) # Reduce to match size 
            
        # Ensure that every lower-layer node gets assigned at least one higher-layer node
        remaining_high_nodes = set(high_nodes)
        assigned_high_nodes = set()    
        
        #guarantee at least one match for each low_node
        for low_node in low_nodes:
            if not remaining_high_nodes:
                break  # Stop if no more high nodes are left
            # Find the closest remaining high node
            closest_high = min(remaining_high_nodes, key=lambda hn: low_node.location.distance(hn.location))
            node_mappings[area_key][low_node] = [closest_high]
            assigned_high_nodes.add(closest_high)
            remaining_high_nodes.remove(closest_high)
            
            used_high_nodes.add(closest_high)
        
        # Assign remaining high nodes optimally while avoiding duplicate assignments
        for high_node in high_nodes:
            if high_node not in assigned_high_nodes:
                # Find the closest low node that still has only one high node
                closest_low = min(
                    node_mappings[area_key].keys(),
                    key=lambda ln: ln.location.distance(high_node.location),
                )
                node_mappings[area_key][closest_low].append(high_node)
                assigned_high_nodes.add(high_node)
                used_high_nodes.add(closest_high)

                    
    #save the resulting mapping in node_mappings where each layer_low node is the key and the mappes layer_high nodes are the list of mapped nodes to this layer_low node 
        #only add layer_low nodes to mapping with at least one mapped layer_high node  
  
    connection_type = layer_config.get("intra_layer")
    bandwidth = layer_config.get("capacities")
    #for each layer_low node (key) in node_mappings, get the list of nodes which are mapped to the key layer_low node 
    for area_key, mappings in node_mappings.items():
        for key_node, mapped_nodes in mappings.items():
            node_list = [key_node] + mapped_nodes
            if connection_type == "ring":
                connect_ring_fast(topo, node_list, bandwidth)
            elif connection_type == "star":
                connect_star(topo, key_node, mapped_nodes, bandwidth)
            elif connection_type == "full":
                connect_fully_connected(topo, node_list, bandwidth)
    
    return node_mappings, used_high_nodes




def connect_ring_fast(topology, node_list, bandwidth):
    #connect node_list greedy on a ring
    
    if len(node_list) < 2:
        print("Not enough nodes to form a ring.")
        return

    # Step 1: Pick the leftmost node to start
    start_node = min(node_list, key=lambda node: node.location.x)

    # Step 2: Nearest-Neighbor Heuristic for initial path
    unvisited = set(node_list)
    unvisited.remove(start_node)
    ring_order = [start_node]
    current_node = start_node

    while unvisited:
        # Find the closest unvisited node
        next_node = min(unvisited, key=lambda node: current_node.location.distance(node.location))
        unvisited.remove(next_node)
        ring_order.append(next_node)
        current_node = next_node
    #print("done with the greedy part")
    # Step 3: Quick shortcut check to reduce long detours
    for i in range(len(ring_order) - 2):
        if ring_order[i].location.distance(ring_order[i+2].location) < ring_order[i].location.distance(ring_order[i+1].location):
            # Swap node[i+1] with node[i+2] to avoid unnecessary long path
            ring_order[i+1], ring_order[i+2] = ring_order[i+2], ring_order[i+1]

    # Step 4: Create links to form the ring
    for i in range(len(ring_order)):
        node1 = ring_order[i]
        node2 = ring_order[(i + 1) % len(ring_order)]  # Wrap to close the ring
        topology.create_and_add_links(node1, node2, bandwidth)

def connect_star(topology, center_node, node_list, bandwidth):
    for node in node_list:
        topology.create_and_add_links(center_node, node, bandwidth)
        
def connect_fully_connected(topology, node_list, bandwidth):
    # Iterate over all unique pairs of nodes in the node_list
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if i < j:  # Avoid duplicate pairs and self-connections
                topology.create_and_add_links(node1, node2, bandwidth)


def add_redundancy_layer_list(layer_list, redundancy_list):
    #layer_list is a list of lists with one element per layer which will be multiplied and name-postfix added if R>1 in redundancy_list
    redundant_layers = []
    for i, l  in enumerate(layer_list):
        if redundancy_list[i] ==1 : redundant_layers.append(l)
        else:
            curr_layer = []
            for r in range(1, redundancy_list[i]+1):
                orig_nodes = l[0]
                r_nodes = [Switch(f"{node.name}R{r}", Point((node.location.x + (1*(r-1))), node.location.y)) for node in orig_nodes]
                curr_layer.append(r_nodes)            
            redundant_layers.append(curr_layer)
    return redundant_layers
        




def generate_multiple_layer_topology(topo, layer_list, config):

    #for each defined conncetion
    for i in range(0,len(config.get("connect"))):
        #get the correct config 
        current_config = {k: v[i] for k, v in config.items()}
        #get the indices of the layers to connect
        low, high = current_config.get("connect", ([],[]))
        #loop over all redudant layer versions
        for l_low in layer_list[low]:
            for l_high in layer_list[high]:        
                mapping, used_high_nodes = connect_layers_with_config(topo, l_low, l_high, current_config)
                #make sure we don't re-use any unused nodes for later connections
                l_high = used_high_nodes
    return layer_list



def generate_folium_map(output_path, G, df_boundaries, boundary_names, node_identifiers, node_layer_names, epsg=25832):
    # Convert boundary dataframes to EPSG:4326
    df_boundaries_4326 = [df.to_crs(epsg=4326) for df in df_boundaries]

    # Get center of map from the first boundary dataset
    centroid = df_boundaries_4326[0].geometry.centroid
    m = folium.Map(location=[centroid.y.mean(), centroid.x.mean()], zoom_start=10, tiles="cartodbpositron")

    # Create FeatureGroups for boundaries
    boundary_layers = [folium.FeatureGroup(name=name) for name in boundary_names]
    
    # Colors for boundaries (distinct but muted)
    boundary_colors = ["#5a5a5a", "#6a4c8c", "#4b8e4b", "#f4a300", "#6a5b3d", "#00a6a6"]
    
    # Add boundaries to the map
    for layer, df, color in zip(boundary_layers, df_boundaries_4326, boundary_colors):
        for _, row in df.iterrows():
            folium.GeoJson(row.geometry, style_function=lambda x, c=color: {"color": c, "weight": 2, "fillOpacity": 0}).add_to(layer)

    # Create node layers
    num_node_layers = len(node_layer_names)
    node_layers = [folium.FeatureGroup(name=name) for name in node_layer_names]

    # Generate a more muted color gradient for nodes
    node_colors = plt.cm.plasma(np.linspace(0, 0.7, num_node_layers))  # Use cividis (muted colors)

    # Transform coordinates from EPSG input to EPSG:4326
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)

    # Add nodes to appropriate layers
    for node in G.nodes:
        if "origin" in G.nodes[node] and hasattr(G.nodes[node]["origin"], "location"):
            loc = G.nodes[node]["origin"].location
            name = G.nodes[node]["origin"].name
            curr_layer = None
            for i, identifier in enumerate(node_identifiers):
                if identifier in name:
                    curr_layer = node_layers[i]
                    node_color = f"#{int(node_colors[i][0] * 255):02x}{int(node_colors[i][1] * 255):02x}{int(node_colors[i][2] * 255):02x}"
                    break
            if not curr_layer:
                raise Exception(f"Layer assignment broken for node: {node}")

            lon, lat = transformer.transform(loc.x, loc.y)
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.7,
                popup=f"Node: {node}"
            ).add_to(curr_layer)

    # Create edge layers
    edge_layers = {f"{node_layer_names[i]} - {node_layer_names[i+1]}": folium.FeatureGroup(name=f"{node_layer_names[i]} - {node_layer_names[i+1]}")
                   for i in range(len(node_layer_names) - 1)}

    # Add an additional set of edge layers for intra-layer connections (Layer-i to Layer-i)
    for i in range(len(node_layer_names)):
        edge_layers[f"{node_layer_names[i]} - {node_layer_names[i]}"] = folium.FeatureGroup(name=f"{node_layer_names[i]} - {node_layer_names[i]}")

    # Generate a more muted color gradient for edges (darker colors)
    edge_colors = plt.cm.plasma(np.linspace(0, 0.7, len(edge_layers)))  # Start from darker colors

    # Add edges to the map
    for edge in G.edges:
        node1, node2 = edge
        lon1, lat1 = transformer.transform(G.nodes[node1]["origin"].location.x, G.nodes[node1]["origin"].location.y)
        lon2, lat2 = transformer.transform(G.nodes[node2]["origin"].location.x, G.nodes[node2]["origin"].location.y)

        # Assign edge to the correct layer (always assign lower-layer to upper-layer format)
        layer1, layer2 = None, None
        for i, identifier in enumerate(node_identifiers):
            if identifier in node1:
                layer1 = i
            if identifier in node2:
                layer2 = i
        
        if layer1 is not None and layer2 is not None:
            # Handle intra-layer connections
            if layer1 == layer2:
                layer_name = f"{node_layer_names[layer1]} - {node_layer_names[layer1]}"
            else:
                lower_layer, upper_layer = min(layer1, layer2), max(layer1, layer2)
                layer_name = f"{node_layer_names[lower_layer]} - {node_layer_names[upper_layer]}"
        
            if layer_name in edge_layers:
                edge_color = edge_colors[list(edge_layers.keys()).index(layer_name)]
                edge_color_hex = f"#{int(edge_color[0] * 255):02x}{int(edge_color[1] * 255):02x}{int(edge_color[2] * 255):02x}"
        
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=edge_color_hex,
                    weight=2,
                    opacity=0.7
                ).add_to(edge_layers[layer_name])

    # Add only non-empty layers to the map
    for layer in boundary_layers + node_layers + list(edge_layers.values()):
        if len(layer._children) > 0:  # Only add non-empty layers
            layer.add_to(m)

    # Add LayerControl
    folium.LayerControl().add_to(m)

    # Save and open the map
    m.save(output_path)
    print(f"Map saved at: {output_path}")


