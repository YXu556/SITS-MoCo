# reproject all to EPSG:4326
import rasterio
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rasterio.merge import merge


# -------------------------- Config --------------------------- #
year = '2022'
sites = ['Adams', 'Coahoma', 'Garfield', 'Harvey', 'Haskell', 'Randolph']
# sites = ['Randolph']
# ------------------------------------------------------------- #

root = Path(r"..\data\US\GEE")
out_dir = root / "Mosaic"
for site in sites:
    print("===================== {} =====================".format(site))
    s2_pth = root / 'Origin' / site / (site + "_S2_" + year)
    s2_fns = [f for f in s2_pth.glob('*.tif')]

    s2_df = pd.DataFrame({'path': s2_fns})
    s2_df['group_str'] = s2_df['path'].apply(lambda x: (x.stem.split('-')[0]).split('_')[-1])

    s2_mos = out_dir / site / ('images_' + year)
    s2_mos.mkdir(parents=True, exist_ok=True)

    for date, group in tqdm(s2_df.groupby('group_str')):
        img_fns = group['path'].values
        src_files_to_mosaic = []
        for tif_f in img_fns:
            src = rasterio.open(tif_f)
            src_files_to_mosaic.append(src)

        s2_image, s2_transform = merge(src_files_to_mosaic)
        if s2_image[-1].nonzero()[0].size == 0:
            continue
        doy = np.unique(s2_image[-1, s2_image[-1]!=0])[0] + 1
        s2_image = s2_image[:-1]

        out_profile = src.profile.copy()
        out_profile.update({"height": s2_image.shape[1],
                            "width": s2_image.shape[2],
                            "count": s2_image.shape[0],
                            "transform": s2_transform})

        out_fp = s2_mos / ('_'.join(tif_f.stem.split("-")[0].split('_')[:2] + [str(doy)]) + '.tif')
        with rasterio.open(out_fp, "w", **out_profile) as f:
            f.write(s2_image)