import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


root = Path(r'..\data\US\GEE\Unsupervised_Full')
bandnames = ['blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'swir1', 'swir2']
out_dir = Path(r"..\data\US\2019\Unsupervised_Full")
out_dir.mkdir(parents=True, exist_ok=True)


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

bns = bandnames + ['doy']
bns0 = [s+'_0' for s in bns]
c = len(bns)
# statelist = [
# 'Wisconsin',
# 'Virginia',
# 'West Virginia',
# ]
dirs = [f for f in root.glob('*') if f.is_dir() and f.stem]# in statelist]
count_p = 0
count_n = 0
for dir in dirs:
    print(f"======================== {dir.stem} ========================")
    fns = [f for f in dir.glob('*.csv')]
    for fn in tqdm(fns):
        data = pd.read_csv(fn)
        data = data.rename(columns=dict(zip(bns, bns0)))
        columns = []
        for bn in bns:
            cols = [c for c in data.columns if c.split('_')[0] == bn]
            cols.sort(key=lambda x: int(x.split('_')[1]))
            columns.extend(cols)
        t = len(cols)
        data = data[columns]

        # s2_values = data.values.reshape(-1, c, t).transpose(0, 2, 1)
        # valid_ind_t = np.all(s2_values == 0, axis=2)

        for id, row in data.iterrows():
            name = fn.stem.split('_')[0] + str(id + 1).zfill(4)
            out_fn = out_dir / (name + '.csv')

            row_values = row.values.reshape(c, t).T
            valid_ind_t = np.all(~np.isnan(row_values), axis=1)

            if valid_ind_t.sum() < 10:
                count_n += 1
                continue
            count_p += 1
            row_valid = row_values[valid_ind_t]

            img_val = row_valid[:, :-1]
            doy_val = row_valid[:, -1].reshape(-1, 1)
            weight_val = getWeight(img_val).reshape(-1, 1)

            out_df = pd.DataFrame(np.hstack([img_val, weight_val, doy_val]), columns=bandnames + ['weight', 'doy'])
            out_df.to_csv(out_fn, index=False)
print('positive', count_p, '\nnegative: ', count_n)