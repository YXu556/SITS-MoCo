import rasterio
import numpy as np
import pandas as pd
# from tqdm import tqdm
from pathlib import Path
from rasterio.merge import merge


# -------------------------- Config --------------------------- #
year = '2022'
sites = ['Adams', 'Coahoma', 'Garfield', 'Harvey', 'Haskell', 'Randolph']
# sites = ['Harvey']
# ------------------------------------------------------------- #

root = Path(r"..\data\US\GEE")
out_dir = root / "Mosaic"

for site in sites:
    print("===================== {} =====================".format(site))
    cdl_pth = root / 'Origin' / site / (site + "_CDL_" + year)

    cdl_fns = [f for f in cdl_pth.glob('*.tif')]

    src_cdlfiles_to_mosaic = []
    for tif_f in cdl_fns:
        src = rasterio.open(tif_f)
        src_cdlfiles_to_mosaic.append(src)

    cdl_image, cdl_transform = merge(src_cdlfiles_to_mosaic)
    cdl_profile = src.profile.copy()
    cdl_profile.update({"height": cdl_image.shape[1],
                        "width": cdl_image.shape[2],
                        "transform": cdl_transform})

    cdl_mos = out_dir / site
    cdl_mos.mkdir(parents=True, exist_ok=True)
    out_fp = cdl_mos / (site + "_CDL_" + year + '.tif')
    with rasterio.open(out_fp, "w", **cdl_profile) as f:
        f.write(cdl_image)
    print(site, ', Done.')