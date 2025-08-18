import rasterio
from rasterio.features import geometry_mask
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import shapely
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from src.loading.boundaries import get_country_boundaries

from src.config import DATAPATH

WORLD_TIFF_FILENAME = DATAPATH + 'GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif'


def _tif_to_df(file):

    with rasterio.open(file) as src:
        population = src.read(1)
        transform = src.transform

    rows, cols = np.nonzero(population)
    pop_values = population[rows, cols]

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    df = pd.DataFrame({
        'lon': xs,
        'lat': ys,
        'pop': pop_values
    })

    return df 

def tif_to_df(files):

    dfs = []
    for file in files:
        dfs.append(_tif_to_df(file))
    
    df = pd.concat(dfs, ignore_index=True)
    return df

def load_window(lon1, lat1, lon2, lat2):

    with rasterio.open(WORLD_TIFF_FILENAME) as src:
        # Convert geographic coordinates to image (row, col) indices.
        row_tl, col_tl = src.index(lon1, lat2)  # top left pixel
        row_br, col_br = src.index(lon2, lat1)    # bottom right pixel

        # Calculate width and height in pixels.
        width = col_br - col_tl
        height = row_br - row_tl

        window = rasterio.windows.Window(col_tl, row_tl, width, height)
        raster = src.read(1, window=window)
        transform = src.window_transform(window)

    return raster, transform

def load_window_blocked(lon1, lat1, lon2, lat2, block_size=1024):

    with rasterio.open(WORLD_TIFF_FILENAME) as src:
        # Convert coordinates to pixel indices
        row_tl, col_tl = src.index(lon1, lat2)
        row_br, col_br = src.index(lon2, lat1)
        width = col_br - col_tl
        height = row_br - row_tl
        
        # Initialize output array
        raster = np.zeros((height, width), dtype=src.dtypes[0])
        window = rasterio.windows.Window(col_tl, row_tl, width, height)
        transform = src.window_transform(window)
        
        # Read in blocks
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block_height = min(block_size, height - i)
                block_width = min(block_size, width - j)
                block_window = rasterio.windows.Window(col_tl + j, row_tl + i, block_width, block_height)
                raster[i:i+block_height, j+j+block_width] = src.read(1, window=block_window)
                
    return raster, transform



def load_country_raster(country):
    boundary = get_country_boundaries(country)

    bounding_box = boundary.bounds 
    raster, transform = load_window(*bounding_box)
    mask = geometry_mask(
        [boundary.__geo_interface__],  # Convert shapely shape to GeoJSON-like format
        out_shape=raster.shape,
        transform=transform,
        all_touched=True,  # Only include cells whose center is within the polygon
            invert=False
    )
    raster[mask] = 0

    return raster, transform 

def filter_polys(polys, min_area_threshold=0.1, large_poly_threshold=1_000, simplify_tol=0.001):
    filtered = []
    for poly in polys:
        poly_area = poly.area * 111 * 111
        if poly_area < min_area_threshold:
            continue 

        if poly_area > large_poly_threshold:
            # Simplify with an appropriate tolerance
            poly = poly.simplify(simplify_tol)

        filtered.append(poly)

    return filtered

def load_country_raster_splitted(country, population_tresh=0, min_area_threshold=0.1):
    boundary = get_country_boundaries(country)
    try:
        polys = [poly for poly in boundary.geoms]
    except AttributeError:
        polys = [boundary]
    polys = filter_polys(polys, min_area_threshold)
    bboxes = [poly.bounds for poly in polys]
    rasters = []
    transforms = []
    communities_skiped = 0
    for poly, bounds in tqdm(zip(polys, bboxes), total=len(polys), desc="Processing polygons"):
        raster, transform = load_window(*bounds)
        if raster.size <= 1:
            continue
        
        mask = geometry_mask(
            [poly.__geo_interface__],  # Convert shapely shape to GeoJSON-like format
            out_shape=raster.shape,
            transform=transform,
            all_touched=True,  # Only include cells whose center is within the polygon
                invert=False
        )
        raster[mask] = 0
        population = raster.sum()
        if population == 0:
            continue
        if population < population_tresh:
            communities_skiped += 1
            continue
        
        rasters.append(raster)
        transforms.append(transform)


    print("Communities skipped: ", communities_skiped)
    return rasters, transforms

def load_country_raster_mainland(country, population_thresh=1_000, **kwargs):

    rasters, transforms = load_country_raster_splitted(country, population_tresh=population_thresh, **kwargs)
    raster, transform = max(zip(rasters, transforms), key=lambda x : x[0].sum())
    return raster, transform

def process_single_window(poly_id, chunk_bounds, file_path):
    """Process a single window and return the raster data."""
    try:
        with rasterio.open(file_path, sharing=True) as src:
            minx, miny, maxx, maxy = chunk_bounds
            
            # Get pixel coordinates
            row_tl, col_tl = src.index(minx, maxy)
            row_br, col_br = src.index(maxx, miny)
            
            width = col_br - col_tl
            height = row_br - row_tl
            
            if width <= 0 or height <= 0:
                return None
            
            window = rasterio.windows.Window(col_tl, row_tl, width, height)
            raster = src.read(1, window=window)
            transform = src.window_transform(window)
            
            return {
                'poly_id': poly_id,
                'raster': raster,
                'transform': transform,
                'window': (row_tl, col_tl, height, width)
            }
            
    except Exception as e:
        print(f"Error processing window: {e}")
        return None

def load_country_raster_splitted_mp(country, preview=False, population_tresh=0, min_area_threshold=0.1, num_workers=None, block_size_factor=4, use_memmap=False):
    # Set default number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    
    # Get country boundaries
    boundary = get_country_boundaries(country)
    polys = [poly for poly in boundary.geoms]
    polys = filter_polys(polys, min_area_threshold=min_area_threshold)
    bboxes = [poly.bounds for poly in polys]
    
    # Create tasks for each polygon/chunk
    tasks = []
    polygon_info = {}
    
    # Open the raster file to get resolution and prepare chunks
    with rasterio.open(WORLD_TIFF_FILENAME) as src:
        # Get internal tile structure
        optimal_block_size = src.block_shapes[0]
        # Use multiples of the internal block size
        processing_block_size = optimal_block_size[0] * block_size_factor

        for poly_id, (poly, bbox) in enumerate(zip(polys, bboxes)):
            minx, miny, maxx, maxy = bbox
            
            # Calculate pixel dimensions
            row_tl, col_tl = src.index(minx, maxy)
            row_br, col_br = src.index(maxx, miny)
            width_px = abs(col_br - col_tl)
            height_px = abs(row_tl - row_br)
            
            polygon_info[poly_id] = {
                'poly': poly,
                'bbox': bbox,
                'shape': (height_px, width_px),
                'origin': (row_tl, col_tl)
            }
            
            # If small enough, process as a single window
            if width_px <= processing_block_size and height_px <= processing_block_size:
                tasks.append((poly_id, bbox, WORLD_TIFF_FILENAME))
            else:
                # Split into multiple windows
                for i in range(0, height_px, processing_block_size):
                    for j in range(0, width_px, processing_block_size):
                        # Calculate geographic coordinates for this window
                        chunk_minx, chunk_maxy = src.xy(row_tl + i, col_tl + j)
                        chunk_maxx, chunk_miny = src.xy(
                            min(row_tl + i + processing_block_size, row_br),
                            min(col_tl + j + processing_block_size, col_br)
                        )
                        
                        chunk_bounds = (chunk_minx, chunk_miny, chunk_maxx, chunk_maxy)
                        tasks.append((poly_id, chunk_bounds, WORLD_TIFF_FILENAME))

    print(f"Processing {len(polys)} polygons split into {len(tasks)} windows")
    
    # Process windows in parallel with progress bar
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_window, *task) for task in tasks]
        
        # Process results with progress bar as they complete
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing windows"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Rest of function remains the same
    # Group results by polygon ID
    grouped_results = {}
    for result in results:
        poly_id = result['poly_id']
        if poly_id not in grouped_results:
            grouped_results[poly_id] = []
        grouped_results[poly_id].append(result)
    
    # Continue with the existing implementation...
    # Reconstruct and process each polygon
    rasters = []
    transforms = []
    communities_skiped = 0
    
    total_chunks = sum(len(chunks) for chunks in grouped_results.values())
    progress_bar = tqdm(total=total_chunks, desc="Reconstructing chunks")
    
    # Process polygons in batches for more efficient memory use
    batch_size = 5  # Process 5 polygons at a time
    poly_ids = list(grouped_results.keys())
    
    for i in range(0, len(poly_ids), batch_size):
        batch_ids = poly_ids[i:i+batch_size]
        batch_rasters = []
        batch_transforms = []
        
        for poly_id in batch_ids:
            poly = polygon_info[poly_id]['poly']
            
            # If only one chunk, use it directly
            if len(grouped_results[poly_id]) == 1:
                chunk = grouped_results[poly_id][0]
                raster = chunk['raster']
                transform = chunk['transform']
            else:
                # Stitch multiple chunks together
                height, width = polygon_info[poly_id]['shape']
                row_tl, col_tl = polygon_info[poly_id]['origin']
                
                # Create empty raster for the full polygon
                with rasterio.open(WORLD_TIFF_FILENAME) as src:
                    dtype = src.dtypes[0]
                    if use_memmap and height * width * 4 > 1e9:  # If array will be larger than 1GB
                        import tempfile
                        with tempfile.NamedTemporaryFile() as temp_file:
                            full_raster = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=(height, width))
                            # Create transform for the full raster
                            main_transform = src.transform
                            full_transform = rasterio.transform.Affine(
                                main_transform.a, main_transform.b, main_transform.c + col_tl * main_transform.a,
                                main_transform.d, main_transform.e, main_transform.f + row_tl * main_transform.e
                            )
                            # Place each chunk in its position
                            full_raster = stitch_chunks(grouped_results[poly_id], height, width, row_tl, col_tl, dtype)
                    else:
                        full_raster = np.zeros((height, width), dtype=dtype)
                        # Create transform for the full raster
                        main_transform = src.transform
                        full_transform = rasterio.transform.Affine(
                            main_transform.a, main_transform.b, main_transform.c + col_tl * main_transform.a,
                            main_transform.d, main_transform.e, main_transform.f + row_tl * main_transform.e
                        )
                        # Place each chunk in its position
                        full_raster = stitch_chunks(grouped_results[poly_id], height, width, row_tl, col_tl, dtype)
                
                raster = full_raster
                transform = full_transform
            
            # Apply polygon mask
            mask = geometry_mask(
                [poly.__geo_interface__],
                out_shape=raster.shape,
                transform=transform,
                all_touched=True,
                invert=False
            )
            
            raster[mask] = 0
            population = raster.sum()
            
            if population == 0 or population < population_tresh:
                communities_skiped += 1
                continue
            
            batch_rasters.append(raster)
            batch_transforms.append(transform)
        
        # Add batch results to main lists
        rasters.extend(batch_rasters)
        transforms.extend(batch_transforms)
        
        # Replace the per-chunk progress update
        progress_bar.update(len(batch_ids))  # Update once per polygon instead of per chunk

    
    print("Communities skipped: ", communities_skiped)
    return rasters, transforms





def stitch_chunks(chunks, height, width, row_tl, col_tl, dtype):
    # Pre-sort chunks to minimize memory fragmentation
    chunks_sorted = sorted(chunks, key=lambda x: (x['window'][0], x['window'][1]))
    
    # Pre-allocate the array just once
    full_raster = np.zeros((height, width), dtype=dtype)
    
    # Group chunks by row for more efficient placement
    for chunk in chunks_sorted:
        chunk_raster = chunk['raster']
        chunk_row, chunk_col, chunk_height, chunk_width = chunk['window']
        
        # Calculate relative position
        rel_row = chunk_row - row_tl
        rel_col = chunk_col - col_tl
        
        # Place in one operation
        indices_row = slice(rel_row, rel_row+chunk_height)
        indices_col = slice(rel_col, rel_col+chunk_width)
        full_raster[indices_row, indices_col] = chunk_raster
    
    return full_raster
