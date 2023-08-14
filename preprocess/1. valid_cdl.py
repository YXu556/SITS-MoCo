import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- Config --------------------------- #
year = '2022'
sites = ['Adams', 'Coahoma', 'Garfield', 'Harvey', 'Haskell', 'Randolph']
# sites = ['Harvey']
# ------------------------------------------------------------- #
k = 3

root = Path(r"..\data\US\GEE")

total_num = 0
for i, site in enumerate(sites):
    print("===================== {} =====================".format(site))
    cdl_pth = root / 'Mosaic' / site / (site + '_CDL_' + year + '.tif')

    with rasterio.open(cdl_pth) as f:
        CDL = f.read()
        src_profile = f.profile

    cdl = CDL[0].astype('uint8')
    cultivated = CDL[1]  # 1: non-cultivated, 2: cultivated
    confidence = CDL[2]
    w, h = cdl.shape

    # 3: 当前像素和周围八个像素相同, 3: 当前像素和周围3*3的窗口
    a = F.unfold(F.pad(torch.Tensor(cdl.reshape(1, 1, w, h)), (k // 2, k // 2, k // 2, k // 2)), kernel_size=k)[
        0].numpy()
    valid_pixel = np.all(a == a[0], axis=0).reshape(w, h)

    # confidence>95
    valid_pixel *= confidence > 95

    label_conf = cdl * valid_pixel
    conf_conf = confidence * valid_pixel

    out_pth = str(cdl_pth).replace(cdl_pth.stem, cdl_pth.stem+'_valid')
    out_profile = src_profile.copy()
    out_profile.update({"count": 2, "dtype": np.float32})
    with rasterio.open(out_pth, "w", **out_profile) as f:
        f.write(np.array([label_conf, conf_conf]))