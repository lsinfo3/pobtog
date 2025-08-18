import pandas as pd
import geopandas as gpd


from src.loading.raster import load_country_raster_splitted
from src.algorithms.raster_dualtree import RasterDualTree
from src.utils.geoutils import idx_to_lonlat

from src.config import DATAPATH

def load_precomputed_dualtree(country, capacity=5_000, population_thresh=1):

    try:
        file_path = DATAPATH + f'precomputed/dualtree/{country}_{capacity}_{population_thresh}.csv'
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}. Please run the precomputation script first.")

    # grouped = df.groupby('island_id')
    # dfs = [group for _, group in grouped]
    # return dfs
    return df

