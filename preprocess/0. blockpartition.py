import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile
import sys

root = Path(r"..\data\US\Boundary")
root.mkdir(exist_ok=True, parents=True)
sites = ['Garfield', 'Adams', 'Randolph', 'Harvey', 'Coahoma', 'Haskell']

out_region = root / 'regions.shp'
out_blockpartition = root / 'blockpartition.shp'

# shp_dir = [d for d in root.glob('*') if d.is_dir()]
shp_dir = [root / site for site in sites]

# blockpartition
blocks = list()
for shp_d in shp_dir:
    shp_p = shp_d / (shp_d.stem + '_Blocks.shp')
    shp = gpd.read_file(shp_p)
    if shp.crs != 'EPSG:4326':
        shp = shp.to_crs('EPSG:4326')
    shp = shp.rename({'OID_1': 'ID'}, axis=1)
    shp['origin'] = shp_d.stem
    blocks.append(shp[['ID', 'geometry', 'origin']])
blocks = pd.concat(blocks, ignore_index=True)
blocks['ID'] = blocks.index + 1

num_train = blocks.shape[0] // 3 * 2
num_valid = blocks.shape[0] // 6
num_eval = blocks.shape[0] - num_train - num_valid
indices = np.arange(blocks.shape[0])
np.random.shuffle(indices)
train_ids = indices[:num_train]
valid_ids = indices[num_train: num_train + num_valid]
eval_ids = indices[-num_eval:]

blocks['train'] = 0
blocks['valid'] = 0
blocks['eval'] = 0
blocks.loc[train_ids, 'train'] = 1
blocks.loc[valid_ids, 'valid'] = 1
blocks.loc[eval_ids, 'eval'] = 1

names = np.empty(blocks.shape[0], dtype='object')
names[train_ids] = 'train'
names[valid_ids] = 'valid'
names[eval_ids] = 'eval'
blocks['name'] = names

blocks.to_file(out_blockpartition)