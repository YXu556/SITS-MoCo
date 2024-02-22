import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset

import moco.loader
from .datautils import *


class USCrops(Dataset):
    def __init__(self,
                 mode,
                 root,
                 year=2019,
                 sequencelength=70,
                 dataaug=None,
                 useall=False,
                 num=3000,
                 randomchoice=False,
                 interp=False,
                 nclasses=20,
                 seed=111,
                 preload_ram=True
                 ):
        super(USCrops, self).__init__()

        mode = mode.lower()
        assert mode in ['train', 'valid', 'eval']
        assert year in [2019, 2020, 2021]

        self.root = root
        self.year = year
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.interp = interp

        # class mapping
        self.classmapping = root / f"classmapping{nclasses}.csv"
        self.mapping = pd.read_csv(self.classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"]
        self.classname = self.mapping["classname"]
        self.nclasses = len(self.classes)  # Other

        # index
        self.indexfile = root / str(year) / f'{mode}.csv'
        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index.sequencelength > 0]#.set_index("idx")

        # sample
        sample_str = ['train']
        if useall == False:
            # if mode in sample_str:
            #     self.index = self.index.groupby('classid', as_index=False).apply(
            #             lambda x: x.sample(n=num // (self.nclasses + 1) * 2)  if x.shape[0] > num // (self.nclasses + 1) * 2
            #             else x.sample(n=x.shape[0]))
            #     self.index = self.index.sample(n=num)
            # else:
            #     self.index = self.index.sample(n=num)
            self.cache = root / str(year) / 'npy' / f"{mode.capitalize()}_R{num}_Seed{seed}_Class{self.nclasses}.npy"
        else:
            self.cache = root / str(year) / 'npy' / f"{mode.capitalize()}_All.npy"
        self.index = self.index.reset_index(drop=True)

        # cache
        if preload_ram:
            print('Load', mode, 'dataset')
            if self.cache.exists():
                self.load_cached_dataset()
            else:
                self.cache_dataset()
        else:
            print('Load', mode, 'dataset while training')
            self.X_list = None

        self.mean = np.array([[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]])
        self.std = np.array([0.227, 0.219, 0.222, 0.22 , 0.2  , 0.193, 0.192, 0.182, 0.123, 0.106])

        if dataaug is not None:
            self.tempaug = dataaug
        else:
            self.tempaug = None

    def transform(self, x, rc=False, interp=False):
        doy = x[:, -1]
        x = x[:, :10] * 1e-4
        if self.tempaug is not None:
            x = self.tempaug(x)

        weight = getWeight(x)
        x = (x - self.mean) / self.std

        if interp:
            doy_pad = np.linspace(0, 366, self.sequencelength).astype('int')
            x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
            weight_pad = getWeight(x_pad*self.std+self.mean)
            mask = np.ones((self.sequencelength,), dtype=int)
        elif rc:
            # choose with replacement if sequencelength smaller als choose_t
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # must have x_pad, mask, and doy_pad
            x_pad = x[idxs]
            mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]
            weight_pad = weight[idxs]
            weight_pad /= weight_pad.sum()
        else:
            # padding
            x_length, c_length = x.shape

            if x_length <= self.sequencelength:
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_length] = 1

                x_pad = np.zeros((self.sequencelength, c_length))
                x_pad[:x_length, :] = x[:x_length, :]

                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:x_length] = doy[:x_length]

                weight_pad = np.zeros((self.sequencelength,), dtype=float)
                weight_pad[:x_length] = weight[:x_length]
                weight_pad /= weight_pad.sum()

            else:
                idxs = np.random.choice(x.shape[0], self.sequencelength, replace=False)
                idxs.sort()

                x_pad = x[idxs]
                mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]
                weight_pad = weight[idxs]
                weight_pad /= weight_pad.sum()

        return torch.from_numpy(x_pad).type(torch.FloatTensor), \
               torch.from_numpy(mask==0), \
               torch.from_numpy(doy_pad).type(torch.LongTensor), \
               torch.from_numpy(weight_pad).type(torch.FloatTensor)

    def target_transform(self, y):
        return torch.tensor(y, dtype=torch.long)

    def load_cached_dataset(self):
        print("precached dataset files found at " + str(self.cache))
        self.X_list = np.load(self.cache, allow_pickle=True).tolist()
        self.X_list = self.X_list[0]
        if len(self.X_list) != len(self.index):
            print("cached dataset do not match. iterating through csv files.")
            self.cache_dataset()

    def cache_dataset(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.X_list = list()
        for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
            self.X_list.append(pd.read_csv(row.path.replace('0. CurrentWork', '2. PheCo')).values)  # todo your datapath
        self.y_list = self.index.classid.values
        np.save(self.cache, np.vstack([np.array(self.X_list), self.y_list]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]
        y = row['classid']

        if self.X_list is None:
            X = pd.read_csv(row.path).values
        else:
            X = self.X_list[index]

        X = self.transform(X, self.rc, self.interp)
        y = self.target_transform(y)

        return X, y


class MoCoDataset(Dataset):
    def __init__(self,
                 root,
                 year=2019,
                 sequencelength=70,
                 dataaug=None,
                 useall=True,
                 num=3000,
                 randomchoice=False,
                 seed=111
                 ):
        super(MoCoDataset, self).__init__()

        assert year in [2019]

        self.root = root
        self.year = year
        self.sequencelength = sequencelength
        self.rc = randomchoice

        self.mean = np.array([0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188])
        self.std = np.array([0.227, 0.219, 0.222, 0.22 , 0.2  , 0.193, 0.192, 0.182, 0.123, 0.106])

        if dataaug is not None:
            self.dataaug = moco.loader.TwoCropsTransform(dataaug)
        else:
            self.dataaug = None

        # index
        self.dir = root / str(year) / 'Unsupervised'
        self.fns = [f for f in self.dir.glob('*.csv')]

        # sample
        if useall == False:
            idxs = np.random.choice(len(self.fns), num, replace=False)
            self.fns = np.array(self.fns)[idxs].tolist()
            self.cache = root / str(year) / 'npy' / f"Unsupervised_R{num}_Seed{seed}.npy"
        else:
            self.cache = root / str(year) / 'npy' / f"Unsupervised_mini_All.npy"

        # cache
        print('Load unsupervised dataset')
        if self.cache.exists():
            self.load_cached_dataset()
        else:
            self.cache_dataset()

    def transform(self, x):
        doy = x[:, -1]
        x = x[:, :10] * 1e-4  # scale reflectances to 0-1
        x = (x - self.mean) / self.std

        return x, doy

    def load_cached_dataset(self):
        print("precached dataset files found at " + str(self.cache))
        self.X_list = np.load(self.cache, allow_pickle=True).tolist()
        # if len(self.X_list) != len(self.fns):
        #     print("cached dataset do not match. iterating through csv files.")
        #     self.cache_dataset()

    def cache_dataset(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.X_list = list()
        for fn in tqdm(self.fns, desc="loading data into RAM", total=len(self.fns)):
            self.X_list.append(pd.read_csv(fn).values)
        np.save(self.cache, np.array(self.X_list))

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, index):
        X = self.X_list[index]

        X, doy = self.transform(X)
        sample = {
            'x': X,
            'doy': doy,
        }
        if self.dataaug is not None:
            q, k = self.dataaug(sample)

        return q, k


class BERTDataset(Dataset):
    def __init__(self,
                 root,
                 year=2019,
                 sequencelength=70,
                 dataaug=None,
                 useall=True,
                 num=3000,
                 randomchoice=False,
                 clstoken=False,
                 seed=111
                 ):
        super(BERTDataset, self).__init__()

        assert year in [2019, 2020, 2021]

        self.root = root
        self.year = year
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.clstoken = clstoken

        self.mean = np.array([[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]])
        self.std = np.array([0.227, 0.219, 0.222, 0.22 , 0.2  , 0.193, 0.192, 0.182, 0.123, 0.106])

        if dataaug is not None:
            self.dataaug = dataaug
        else:
            self.dataaug = None

        # index
        self.dir = root / str(year) / 'Unsupervised'
        self.fns = [f for f in self.dir.glob('*.csv')]

        # sample
        if useall == False:
            idxs = np.random.choice(len(self.fns), num, replace=False)
            self.fns = np.array(self.fns)[idxs].tolist()
            self.cache = root / str(year) / 'npy' / f"Unsupervised_R{num}_Seed{seed}.npy"
        else:
            self.cache = root / str(year) / 'npy' / f"Unsupervised_All.npy"

        # cache
        print('Load unsupervised dataset')
        if self.cache.exists():
            self.load_cached_dataset()
        else:
            self.cache_dataset()

    def transform(self, x):

        doy = x[:, -1]
        x = x[:, :10] * 1e-4  # scale reflectances to 0-1
        x = (x - self.mean) / self.std

        return x, doy

    def padding(self, ts, doy, rc=False):
        if rc:
            # choose with replacement if sequencelength smaller als choose_t
            ts_length = self.sequencelength
            replace = False if ts.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(ts.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # must have x_pad, mask, and doy_pad
            ts_origin = ts[idxs]
            ts_mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]

            ts_masked, mask = self.random_masking(ts_origin, ts_length)
        else:
            ts_length, dim = ts.shape

            if ts_length <= self.sequencelength:
                ts_mask = np.zeros((self.sequencelength,), dtype=int)
                ts_mask[:ts_length] = 1

                ts_origin = np.zeros((self.sequencelength, dim))
                ts_origin[:ts_length, :] = ts

                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:ts_length] = doy
            else:
                ts_length = self.sequencelength
                idxs = np.random.choice(ts.shape[0], self.sequencelength, replace=False)
                idxs.sort()

                ts_origin = ts[idxs]
                ts_mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]

            ts_masked, mask = self.random_masking(ts_origin, ts_length)

        output = {"bert_input": ts_masked,
                  "bert_target": ts_origin,
                  "bert_mask": ts_mask,
                  "loss_mask": mask,
                  "time": doy_pad,
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}

    def random_masking(self, ts, ts_length):
        ts_masking = ts.copy()
        mask = np.zeros((self.sequencelength,), dtype=int)

        for i in range(ts_length):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                mask[i] = 1

                if prob < 0.5:
                    ts_masking[i, :] += np.random.uniform(low=-0.5, high=0, size=(ts.shape[1],))

                else:
                    ts_masking[i, :] += np.random.uniform(low=0, high=0.5, size=(ts.shape[1],))

        return ts_masking, mask

    def load_cached_dataset(self):
        print("precached dataset files found at " + str(self.cache))
        self.X_list = np.load(self.cache, allow_pickle=True).tolist()
        if len(self.X_list) != len(self.fns):
            print("cached dataset do not match. iterating through csv files.")
            self.cache_dataset()

    def cache_dataset(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.X_list = list()
        for fn in tqdm(self.fns, desc="loading data into RAM", total=len(self.fns)):
            self.X_list.append(pd.read_csv(fn).values)
        np.save(self.cache, np.array(self.X_list))

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        X = self.X_list[index]

        X, doy = self.transform(X)

        return self.padding(X, doy, self.rc)
