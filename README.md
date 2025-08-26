# GeoPopulation Analysis Toolkit

## Installation
git lfs pull
pip install -e . 

## Overview
This toolkit provides advanced algorithms and utilities for processing, analyzing, and visualizing geographic population data. It specializes in creating population-weighted clusters from high-resolution raster datasets using various spatial partitioning techniques.

## Key Components

### Data Sources 
There are two external main data sources for this project:
- One large GeoTiff file (10.37GB) that is a raster dataset of the entire inhabited world in 3 arc seconds resolution (one pixel is represents roughly 100m x 100m) where each pixels value is an estimate for the population living in that pixel.
- Available via: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_4326_3ss/V1-0/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.zip
Can be loaded with rasterio:
```python
with rasterio.open('data/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif') as src:
    raster = src.read(1)
    transform = src.transform
```
The transform object is responsible for mapping the indices of the raster grid back to longitude and latitude coordinates. 



- A .gpkg file that contains administrative boundaries for most countries (263) in the world. The following layers are available: 'ADM_0', ADM_1', 'ADM_2', 'ADM_3', 'ADM_4', 'ADM_5'. Currently only the countries borders  ('ADM_0') are used. 
-Available via: https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip
Can be loaded with geopandas:
```python 
gdf = gpd.read_file('data/gadm_410-levels.gpkg', layer='ADM_0')
```


## Helpers for loading data

### Loading boundaries
To get a countries boundary as a shapely object:
```python
from geopop.loading.boundaries import get_country_boundaries

boundaries = get_country_boundaries('Germany')
```

### Loading rasters
Loads data for a rectangular window defined by geographic coordinates and returns a 2D numpy array and the corresponding transform object
```python
raster, transform = load_window(lon1, lat1, lon2, lat2)
```

Uses the countries borders to call and return the results of load_window
```python
raster, transform = load_country_raster('Germany')
```

Same as above but treats each disconnected polygon (e.g islands) as its own raster and returns a list of rasters and corresponding transforms instead. 
Skips islands with population below population_tresh
```python
rasters, transforms = load_country_raster_splitted('Spain', population_tresh=5000)
```

Same as above but uses multithreading with chunking for faster loading of large countries
```python
rasters, transforms = load_country_raster_splitted_mp(
    'Indonesia', 
    population_tresh=5000,
    min_area_threshold=0.1,  # Skip polygons smaller than this threshold (km²)
    num_workers=None,        # Default: use all CPU cores
    block_size_factor=4,     # Controls memory usage - higher values use more memory but fewer operations
    use_memmap=False         # Set to True for very large countries to reduce memory usage
)
```

Like load_country_raster but only loads the mainland (largest population polygon) of a country
```python
raster, transform = load_country_raster_mainland('Germany', population_thresh=1000)
```

### Dual Tree Algorithm
The core algorithm implemented in `raster_dualtree.py` efficiently partitions population raster data into balanced regions. This algorithm:

- Creates a hierarchical spatial decomposition of population data
- Ensures each region contains approximately equal population
- Supports weighted splitting based on population distribution
- Provides geographic boundary and centroid calculations

### Geographic Utilities
The project includes several utility modules:

#### geoutils.py: Core geographic transformation and visualization functions
- Transform between pixel coordinates and geographic coordinates
- Visualization tools for centroids and clusters
- Formatting utilities for population data

#### geoboundaries.py: Country boundary management
- Download administrative boundaries from geoboundaries.org
- Process and simplify boundary polygons
- Utilities for spatial operations on boundaries

## Data Processing Pipeline
The `prep_world.py` module implements a complete workflow for:

- Loading global population raster data for specific countries
- Processing large datasets with multi-processing optimizations
- Creating population-weighted clusters using the dual tree algorithm
- Visualizing results with interactive Plotly maps

## Features
- **High Performance**: Optimized for processing large geographic datasets
- **Multi-scale Analysis**: Support for various administrative levels
- **Population-weighted Clustering**: Create clusters with balanced population distribution
- **Geographic Visualization**: Interactive maps and visualizations
- **Parallel Processing**: Multi-threaded processing for large datasets

## Usage Examples
```python
# Load population data for a country
rasters, transforms = load_country_raster_splitted('Germany', population_tresh=5000)

# Create population-weighted clusters
centers = dualtree_on_rasters(rasters, transforms, capacity=100_000)

# Visualize results
fig = plot_centroids_from_df(centers)
fig.show()
```


## Web Viewer
A web-based population viewer is available in the `geo-population-viewer` directory, providing an interactive interface for exploring population clusters.

## Performance Considerations
For large datasets, consider using the multiprocessing options and adjusting parameters like `block_size_factor` and `capacity` to optimize performance and memory usage.

## POBTOG
The POBTOG Demo can be found in examples/GUI.ipynb
Depending on the country size and topology settings it is adviced to run the demo on a machine with more than 100GB RAM and multiple cores.
