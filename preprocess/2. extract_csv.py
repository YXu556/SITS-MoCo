import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from rasterio.mask import mask
from shapely.geometry import mapping

# -------------------------- Config --------------------------- #
year = '2022'
sites = ['Adams', 'Coahoma', 'Garfield', 'Harvey', 'Haskell', 'Randolph']
# sites = ['Haskell']
# ------------------------------------------------------------- #
bandnames = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2']

root = Path(r'..\data\US\GEE\Mosaic')
out_dir = Path(r"..\data")
classmapping = Path(r"..\data\classmapping15.csv")
shp_fn = Path(r"..\data\US\Boundary\blockpartition.shp")

cmapping = pd.read_csv(classmapping)
cmapping = cmapping.set_index("code")
classes = cmapping["id"].unique()
classname = cmapping["classname"].unique()
nclasses = len(classes)


def getWeight(x):
    score = np.ones(x.shape[0])
    score = np.minimum(score, (x[:, 0] / 10000 - 0.1) / 0.4)  # blue
    score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) / 10000 - 0.2) / 0.6)  # rgb
    cloud = score * 100 > 20  # todo

    dark = x[:, [6, 8, 9]].sum(1) < 3500  # todo

    ndvi = (x[:, 6] - x[:, 2]) / (x[:, 6] + x[:, 2] + 1e-8)
    ndvi[cloud] = -1
    ndvi[dark] = -1

    weight = np.exp(ndvi)
    weight /= weight.sum()

    return weight

# open labels
labels = {}
indices = {}
print('=================== Open CDL ===================\n')
for i, site in enumerate(sites):
    cdl_pth = root / site / (site + '_CDL_' + year + '_valid.tif')
    cdl_dataset = rasterio.open(cdl_pth)
    labels[site] = cdl_dataset
    indices[site] = list()
print('Done.\n')

# read images and save csv
images = {}
doys = {}
print('=================== Open images ===================')
for site in sites:
    print("------------ {} -----------".format(site))
    img_pth = root / site / ('images_' + year)

    img_fns = [f for f in img_pth.glob('*.tif')]
    img_fns.sort(key=lambda x: int(x.stem.split('_')[-1]))
    doy = np.array([int(img_fn.stem.split('_')[-1]) for img_fn in img_fns])

    # read images
    data = []
    for img_fn in tqdm(img_fns):
        with rasterio.open(img_fn) as f:
            s2_image = f.read().transpose(1, 2, 0)
        data.append(s2_image)
    images[site] = np.array(data)
    doys[site] = doy

# traverse blocks
blockpartition = gpd.read_file(shp_fn)
if blockpartition.crs != 'EPSG:4326':
    blockpartition = blockpartition.to_crs('EPSG:4326')

pidx = 0
pindices = list()
print('=================== Convert to csv ===================')
for i, (idx, row) in enumerate(blockpartition.iterrows()):
    print(f'-------- Block {row.ID} / {blockpartition.shape[0]} ---------')
    geom = row.geometry
    feature = [mapping(geom)]
    site = row.origin
    id = row.ID
    mode = row['name']
    if site not in labels.keys():
        continue
    out_dir_s = out_dir / year / site
    out_dir_s.mkdir(parents=True, exist_ok=True)

    out_label, out_transform = mask(labels[site], feature, crop=False)
    out_conf = out_label[1].reshape(1, *out_label.shape[1:])
    out_label = out_label[0].reshape(1, *out_label.shape[1:])

    # images
    s2_image = images[site]
    doy = doys[site]
    t, w, h, c = s2_image.shape

    # screening according to the valid labels
    valid_ind = out_label > 0
    valid_lb = out_label[valid_ind]
    valid_conf = out_conf[valid_ind]
    num_valid = (valid_ind > 0).sum()
    valid_ind = np.repeat(valid_ind, t, axis=0)
    valid_s2 = s2_image[valid_ind].reshape(t, -1, c)
    valid_s2 = valid_s2.transpose(1, 0, 2).astype(float)

    # screening according to length of time
    valid_ind_t = np.all(valid_s2 == 0, axis=2)
    valid_s2[valid_ind_t] = np.nan

    for pid in tqdm(range(num_valid)):
        name = str(id) + str(pid + 1).zfill(5)
        out_fn = out_dir_s / (name + '.csv')

        img = valid_s2[pid]
        lb = int(valid_lb[pid])
        conf = int(valid_conf[pid])

        img_val_ind = np.all(~np.isnan(img), axis=1)
        img_val = img[img_val_ind]
        # weight_val = getWeight(img_val).reshape(-1, 1)
        doy_val = doy[img_val_ind].reshape(-1, 1)

        out_df = pd.DataFrame(np.hstack([img_val, doy_val]), columns=bandnames + ['doy'])
        out_df.to_csv(out_fn, index=False)

        if lb in cmapping.index:
            classid = cmapping.loc[lb, 'id']
            classname = cmapping.loc[lb, 'classname']
        else:
            classid = nclasses
            classname = 'Other'
        pindex = {'idx': pidx, 'id': name, 'code': lb, 'confidence': conf, 'path': str(out_fn), 'sequencelength': img_val.shape[0],
                  'classid': classid, 'classname': classname, 'region': site, 'mode': mode}
        pindices.append(pindex)
        pidx += 1

out_fn = str(out_dir / year / 'index.csv')
index_df = pd.DataFrame(pindices).set_index('idx')
index_df.to_csv(out_fn)
