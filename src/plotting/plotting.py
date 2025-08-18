import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import shapely
from random import shuffle, randint
import mapclassify

from src.utils.utils import format_big_number
from src.loading.precomputed import load_precomputed_dualtree
from src.algorithms.hierarchy import kmeans_tree_from_points
from src.algorithms.tsp import solve_tsp_large
from src.algorithms.raster_dualtree import RasterDualTree
from src.loading.raster import load_country_raster_splitted
from src.utils.geoutils import idx_to_lonlat

import numpy as np

from .helpers import affine_normalize


def plot_hierarchy_tree(df, fig=None, point_size=5, line_width=0.5, colorscale='jet', **kwargs):
    """
    Create a scattermap plot showing all nodes and their parent connections.
    
    Args:
        df: DataFrame containing 'lon', 'lat', 'id', 'parent_id' columns
        fig: Optional existing figure to add the traces to
        point_size: Size of the points
        line_width: Width of the connection lines
        **kwargs: Additional keyword arguments for scatter trace
        
    Returns:
        A plotly Figure object
    """
    df = df.copy()

    if fig is None:
        fig = go.Figure()

    max_level = df['level'].max()
    colors = px.colors.sample_colorscale(colorscale, np.linspace(0, 0.6, max_level+1))
    
    for level in df['level'].unique():
        mask = df['level'] == level
        df.loc[mask, 'point_color'] = colors[level]
        df.loc[mask, 'point_size'] = (max_level - level) ** 2 + point_size

    # Plot all points
    #fig.add_trace(
    #    go.Scattermap(
    #        lon=df['lon'],
    #        lat=df['lat'],
    #        mode='markers',
    #        marker=dict(size=df['point_size'], color=df['point_color']),
    #        hoverinfo='text',
    #        hovertext=[f"ID: {id_val}<br>Pop: {pop}" for id_val, pop in zip(df['id'], df['pop'])],
    #        name='Nodes',
    #        **kwargs
    #    )
    #)
    
    # Create parent-child connections
    # Filter out nodes without parents
    child_nodes = df[~df['parent_id'].isna()].copy()
    
    if len(child_nodes) > 0:
        # Create a mapping of parent IDs to their coordinates and levels
        parent_coords = df.set_index('id')[['lon', 'lat', 'level']]
        
        # Merge child nodes with parent coordinates
        child_nodes = child_nodes.merge(
            parent_coords,
            left_on='parent_id',
            right_index=True,
            suffixes=('', '_parent')
        )
        
        # Calculate level difference between child and parent
        child_nodes['level_diff'] = child_nodes['level_parent'] - child_nodes['level']
        
        # Define colors for different level differences
        level_diff_colors = px.colors.sample_colorscale('Greys', np.linspace(0.7, 1, len(child_nodes['level_diff'].unique())))
        linestyles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']  # Add more styles if needed
        
        # Group by level difference
        level_diff_groups = child_nodes.groupby('level_diff')
        for level_diff, group in level_diff_groups:
            # Select color based on level difference
            color_idx = abs(int(level_diff)) % len(level_diff_colors)
            line_color = level_diff_colors[color_idx]
            linestyle = linestyles[abs(int(level_diff)) % len(linestyles)]
            
            # Adjust line width based on level difference
            adjusted_width = line_width * (0.8 + 0.4 * abs(int(level_diff)))
            
            lons = []
            lats = []
            
            # Build arrays of coordinates with None separators between line segments
            for _, row in group.iterrows():
                lons.extend([row.lon, row.lon_parent, None])
                lats.extend([row.lat, row.lat_parent, None])
            
            fig.add_trace(
                go.Scattermap(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(
                        width=adjusted_width,
                        color=line_color,
                        # dash=linestyle
                    ),
                    hoverinfo='none',
                    name=f'Level diff: {abs(int(level_diff))}',
                    legendgroup=f'level_diff_{level_diff}',
                )
            )

    # Plot all points
    fig.add_trace(
        go.Scattermap(
            lon=df['lon'],
            lat=df['lat'],
            mode='markers',
            marker=dict(size=df['point_size'], color=df['point_color']),
            hoverinfo='text',
            hovertext=[f"ID: {id_val}<br>Pop: {pop}" for id_val, pop in zip(df['id'], df['pop'])],
            name='Nodes',
            **kwargs
        )
    )
    
    # Set map layout
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox=dict(
            bearing=0,
            pitch=0
        ),
        showlegend=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

def plot_tsp_links(df, tsp_edges, fig=None, line_width=1.5, line_color='rgb(255, 50, 50)', **kwargs):

    """
    Adds TSP solution edges to a Plotly Scattermap figure with dotted lines.
    
    Args:
        df: DataFrame containing 'lon', 'lat' columns
        tsp_edges: 2D numpy array where each row contains two indices representing
                  the source and target points in the TSP solution
        fig: Optional existing figure to add the traces to
        line_width: Width of the lines
        line_color: Color of the lines
        **kwargs: Additional keyword arguments for the scatter trace
        
    Returns:
        A plotly Figure object with the added TSP links
    """
    if fig is None:
        fig = go.Figure()
    
    if len(tsp_edges) == 0:
        return fig
    
    # Prepare coordinates for the line segments
    lons = []
    lats = []
    
    # Extract coordinates for each segment in the TSP solution
    for src_idx, tgt_idx in tsp_edges:
        src_lon = df.iloc[src_idx]['lon']
        src_lat = df.iloc[src_idx]['lat']
        tgt_lon = df.iloc[tgt_idx]['lon']
        tgt_lat = df.iloc[tgt_idx]['lat']
        
        # Add the line segment with None separator
        lons.extend([src_lon, tgt_lon, None])
        lats.extend([src_lat, tgt_lat, None])
    
    # Add the trace with dotted lines
    fig.add_trace(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(
                width=line_width,
                color=line_color,
                            ),
            hoverinfo='none',
            name='TSP Route',
            **kwargs
        )
    )
    
    return fig


def plot_levels(df, fig=None, scale=2, **kwargs):

    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(name='|'))
        fig.add_trace(go.Scatter(name='|'))
    
    groups = sorted(df.groupby('level'), key=lambda x: x[0], reverse=True)
    colors = px.colors.sample_colorscale('thermal', [i/(len(groups)-1) for i in range(len(groups))])
    shuffle(colors)
    for i, (level, group) in enumerate(groups):
        # if i % 2 == 0:
        #     continue
        print('group size: ', len(group))
        fig = plot_centroids_from_df(group, scale=scale+1*(len(groups) - level), color=colors[i], name=f'level {group["level"].values[0]}', logsize=True, fig=fig, visible='legendonly', **kwargs)
    
    return fig

def plot_regions(df, level=-1, region_level=1, fig=None, scale=3, **kwargs):
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(name='|'))
        fig.add_trace(go.Scatter(name='|'))

    if level == -1:
        try:
            filtered_df = df[df['is_leaf'] == True].copy()
        except KeyError:
            filtered_df = df[df['level'] == max(df['level'].unique())].copy()
    else:
        filtered_df = df[df['level'] == level].copy()

    if len(filtered_df) == 0:
        return fig

    # Get unique regions and create color mapping
    regions = filtered_df.groupby(f'ancestor{region_level}')
    region_ids = list(regions.groups.keys())

    # Check if DataFrame has geometry column and use mapclassify.greedy if it does
    if hasattr(df, 'geometry'):
    
        region_gdf = df[df['level'] == region_level].copy()
        
        if len(region_gdf) > 1:
            region_gdf['color'] = mapclassify.greedy(region_gdf, min_colors=10, min_distance=1)
            num_colors = len(region_gdf['color'].unique())
            
            if num_colors <= 10:
                palette = px.colors.qualitative.G10[:num_colors]
            else:
                palette = px.colors.qualitative.Bold[:num_colors] if num_colors <= 20 else \
                         [f'rgb({randint(0, 255)}, {randint(0, 255)}, {randint(0, 255)})' for _ in range(num_colors)]
            # palette = px.colors.sample_colorscale('thermal', [i/(num_colors-1) for i in range(num_colors)])
            
            region_gdf['color'] = region_gdf['color'].map(lambda x: palette[x])
            
            filtered_df['color'] = 'rgb(150, 150, 150)' 
            for i, region_id in enumerate(region_ids):
                filtered_df.loc[filtered_df[f'ancestor{region_level}'] == region_id, 'color'] = region_gdf.loc[region_gdf['id'] == region_id, 'color'].values[0]
        else:
            # Not enough regions for greedy coloring
            raise ValueError("Not enough regions for greedy coloring")
                
    else:
        # Generate random colors as before (if no geometry column)
        colors = [f'rgb({randint(0, 255)}, {randint(0, 255)}, {randint(0, 255)})' for _ in range(len(region_ids))]
        color_array = np.zeros(len(filtered_df), dtype=object)
        for i, (region_id, indices) in enumerate(regions.indices.items()):
            color_array[indices] = colors[i]

    # Call plot_centroids_from_df just once with the color array
    fig = plot_centroids_from_df(filtered_df, scale=scale, fig=fig, 
                                color=filtered_df['color'], 
                                name=f'level {level}', 
                                logsize=True, **kwargs)

    return fig

def plot_polys_colored(gdf, colorscale='thermal', opacity=0.7, fig=None):

    gdf = gdf.copy()
    gdf['color'] = mapclassify.greedy(gdf, min_colors=8)
    
    # Create a color map based on the colorscale
    unique_colors = gdf['color'].nunique()
    colors = px.colors.sample_colorscale(colorscale, unique_colors)
    
    # Group all geometries by color value
    traces = []
    for color_idx in range(unique_colors):
        # Filter polygons of this color
        color_group = gdf[gdf['color'] == color_idx]
        
        # Convert all polygons of this color to a single multipolygon
        multi_poly = color_group.union_all()
        
        
        # Extract all exterior coordinates with None separators
        coords = []
        lons = []
        lats = []
        
        # Process polygon or multipolygon
        if multi_poly.geom_type == 'Polygon':
            exterior_coords = list(multi_poly.exterior.coords)
            lons, lats = zip(*exterior_coords)
        else:  # MultiPolygon
            for poly in multi_poly.geoms:
                # Get this polygon's coordinates
                if poly.geom_type == 'LineString':
                    continue
                exterior_coords = list(poly.exterior.coords)
                poly_lons, poly_lats = zip(*exterior_coords)
                
                # Add to our lists with None separators
                lons += list(poly_lons) + [None]
                lats += list(poly_lats) + [None]
            
            # Remove the last None if it exists
            if lons and lons[-1] is None:
                lons.pop()
                lats.pop()
        
        # Create a trace for this color group
        traces.append(
            go.Scattermap(
                lon=lons,
                lat=lats,
                mode='lines',
                fill='toself',
                fillcolor=colors[color_idx],
                opacity=opacity,
                marker={'color': colors[color_idx], 'size': 0},
                line={'width': 1, 'color': colors[color_idx]},
                # name=f'Region {color_idx}',
                showlegend=False,
                hoverinfo='text',
                hovertext= [f"{format_big_number(p)}" for p in color_group['pop'].values]
            )
        )
    
    # Create figure with all traces
    if fig is None:
        fig = go.Figure()
    
    fig.add_traces(traces)
    # fig = go.Figure(traces)
    
    # Calculate center for the map view
    center_lat = gdf['lat'].mean()
    center_lon = gdf['lon'].mean()
    
    # Update layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig


def plot_boundaries(bounds, fig=None):
    if fig is None:
        fig = go.Figure()
    for poly in bounds.geoms:
        lon_outline, lat_outline = poly.exterior.coords.xy
        fig.add_trace(go.Scattermap(
            lon=list(lon_outline),
            lat=list(lat_outline),
            mode='lines',
            line=dict(color='black', width=1),
            hoverinfo='skip'
        ))
    return fig 

def plot_centroids_from_df(df, **kwargs):
    return plot_centroids(df[['lon', 'lat']].values, pop=df['pop'], **kwargs)

def plot_centroids(centroids, pop, scale=100, fig=None, color=None, logsize=False, **kwargs):
    if len(centroids) != len(pop):
        raise ValueError('not same length')
    
    # Convert pop to numpy array and ensure it's a float type
    pop = np.asarray(pop, dtype=np.float64)
    
    # Calculate scaled sizes
    scaled_sizes = pop / pop.mean() if len(pop) > 1 else np.array([1.0])
    
    if logsize and len(pop) > 1:
        # Only apply log transform if we have multiple points
        try:
            # Safe log transform
            min_val = np.min(scaled_sizes[scaled_sizes > 0]) if np.any(scaled_sizes > 0) else 1e-10
            log_sizes = np.log(scaled_sizes + min_val)
            scaled_sizes = log_sizes * (scaled_sizes.std() or 1.0)  # Use 1.0 if std is 0
        except Exception as e:
            print(f"Warning: Could not apply log scaling: {e}")
    
    # Apply final scaling
    scaled_sizes = (2 + affine_normalize(scaled_sizes)) * scale

    if fig is None:
        fig = go.Figure()

    # Format population numbers with underscores
    formatted_pop = [f"{p:_}" for p in pop]

    fig.add_trace(go.Scattermap(
        lon=centroids[:, 0],
        lat=centroids[:, 1],
        mode='markers',
        marker=dict(
            size=scaled_sizes,
            color=scaled_sizes if color is None else color,
        ),
        hovertext=formatted_pop,
        **kwargs 
    ))

    # Rest of your function...
    return fig

def plot_fitted_hist(x, fit=True, fig=None, **kwargs):
    if fig is None:
        fig = go.Figure()

    print(f'mean: {int(x.mean()):_}, \n std: {int(x.std()):_} \n variation: {round(x.std() / x.mean(), 4)}')

    # fig = px.histogram(x, histnorm='probability density' if fit else None, **kwargs)
    fig.add_trace(go.Histogram(x=x, histnorm='probability density' if fit else None, **kwargs))

    if fit:
        from scipy import stats
        shape, loc, scale = stats.lognorm.fit(x)

        linspace = np.linspace(x.min(), x.max(), 2_000)

        fig.add_trace(go.Scatter(x=linspace, y=stats.lognorm.pdf(linspace, shape, loc, scale), mode='lines', name='Fitted Lognormal'))
    return fig

def plot_net_precomputed(countries, fig=None, capacity=5_000, population_thresh=30_000):
    if fig is None:
        fig = go.Figure()

    for country in countries:
        print('Processeing', country)
        core_nodes_all = None

        dfs = load_precomputed_dualtree(country, capacity=capacity, population_thresh=population_thresh)

        island_dfs = (df for _, df in dfs.groupby('island_id'))
        for df in island_dfs:

            n_clusters = [max(round(len(df) * 0.1), 1)]
            if n_clusters[0] >= 1:
                n_clusters += [max(round(n_clusters[-1] * 0.02), 1)]


            
            dfk_ij = kmeans_tree_from_points(df[['lon', 'lat', 'pop']].values, n_clusters=n_clusters)
            fig = plot_hierarchy_tree(dfk_ij, fig=fig)

            core_nodes = dfk_ij[dfk_ij['level'] == 0]
            if core_nodes is None:
                core_nodes_all = core_nodes 
            else:
                core_nodes_all = pd.concat([core_nodes_all, core_nodes], axis=0)

        if core_nodes_all is not None and len(core_nodes_all) > 1:
            print('starting tsp with ', len(core_nodes_all), 'nodes')
            core_links = solve_tsp_large(core_nodes_all[['lon', 'lat']].values, max_points=1_000)
            fig = plot_tsp_links(core_nodes_all, core_links, fig=fig, line_color='black')
    
    return fig 

def plot_net(countries, fig=None, capacity=5_000, population_thresh=30_000):
    if fig is None:
        fig = go.Figure()

    for country in countries:
        print('Processeing', country)
        core_nodes_all = None

        rasters, transforms = load_country_raster_splitted(country, population_tresh=population_thresh)

        for raster, transform in zip(rasters, transforms):
            population = raster.sum()
            # capacity = 5_000 +  (population // 10_000)
            n_clusters = [max(round(population//capacity * 0.1), 2)]
            n_clusters += [max(round(n_clusters[-1] * 0.02), 1)]

            centers = RasterDualTree.build_from_raster(raster, capacity=capacity, easy_split=False) 
            idx_to_lonlat(centers, transform, inplace=True)
            dfk_ij = kmeans_tree_from_points(centers, n_clusters=n_clusters)
            fig = plot_hierarchy_tree(dfk_ij, fig=fig)

            core_nodes = dfk_ij[dfk_ij['level'] == 0]
            if core_nodes is None:
                core_nodes_all = core_nodes 
            else:
                core_nodes_all = pd.concat([core_nodes_all, core_nodes], axis=0)

        if core_nodes_all is not None and len(core_nodes_all) > 1:
            print('starting tsp with ', len(core_nodes_all), 'nodes')
            core_links = solve_tsp_large(core_nodes_all[['lon', 'lat']].values, max_points=1_000)
            fig = plot_tsp_links(core_nodes_all, core_links, fig=fig, line_color='black')
    
    return fig 