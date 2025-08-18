import pandas as pd
import geopandas as gpd


from geopop.loading.raster import load_country_raster_splitted
from geopop.algorithms.raster_dualtree import RasterDualTree
from geopop.utils.geoutils import idx_to_lonlat

from src.config import DATAPATH


def precompute_country(country, capacity=1_000, population_thresh=1, use_quantile=True, **kwargs):
    rasters, transforms = load_country_raster_splitted(country, population_tresh=population_thresh)

    df = None
    id = 0
    rasters, transforms = zip(*sorted(zip(rasters, transforms), key=lambda r: r[0].sum(), reverse=True))
    for raster, transform in zip(rasters, transforms):
        id += 1
        centers = RasterDualTree.build_from_raster(raster, capacity=capacity, easy_split=False, **kwargs) 
        idx_to_lonlat(centers, transform, inplace=True)

        df_sub = pd.DataFrame(centers, columns=['lon', 'lat', 'pop'])
        df_sub['island_id'] = id
        if df is None:
            df = df_sub
        else:
            df = pd.concat([df, df_sub], axis=0)

    if use_quantile:
        df.to_csv(DATAPATH + f'precomputed/qdualtree/{country}_{capacity}_{population_thresh}.csv', index=False)
    else:
        df.to_csv(DATAPATH + f'precomputed/dualtree/{country}_{capacity}_{population_thresh}.csv', index=False)



if __name__ == "__main__":

    COUNTRIES = ['Indonesia',
        'USA',
        'Japan',
        'Taiwan',
        'Germany',
        'Italy',
        'France',
        'South Africa',
        'Austria',
        'GBR',
        'Nigeria']
    
    for country in COUNTRIES:
        print(f"Precomputing {country}...")
        precompute_country(country, capacity=10_000, population_thresh=30_000, use_quantile=False, alpha=0.3)

        