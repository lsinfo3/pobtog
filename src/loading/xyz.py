import os 
import pandas as pd

DATAPATH = '/data'



def load_precomputed_dualtree_precomputed(country, capacity=1_000):

    df = pd.read_csv(f'precompute/dualtree/{country}_{capacity}.csv')

    grouped = df.groupby('island_id')
    return [group for _, group in grouped]