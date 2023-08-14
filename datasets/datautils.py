"""
This script is for data augmentation
"""
import random
import numpy as np

import torch


# ---------------------------- Spectral augmentation ----------------------------
class RandomChanSwapping:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            idx[s_idx[1]] = s_idx[0]
            x = x[:, idx]
        return x


class RandomChanRemoval:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            x = x[:, idx]
        return x


class RandomAddNoise:
    def __call__(self, sample):
        if isinstance(sample, dict):
            x = sample['x']
        else:
            x = sample['x']
        t, c = x.shape
        for i in range(t):
            prob = np.random.rand()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.5:
                    x[i, :] += -np.abs(np.random.randn(c)*0.5)#np.random.uniform(low=-0.5, high=0, size=(c,))
                else:
                    x[i, :] += np.abs(np.random.randn(c)*0.5)#np.random.uniform(low=0, high=0.5, size=(c,))
        if isinstance(sample, dict):
            return {'x': x, 'doy': sample['doy']}
        else:
            return x


# ---------------------------- Temporal augmentation ----------------------------
class RandomTempSwapping:
    def __call__(self, x):
        if np.random.rand() < 0.5:
            s_idx = random.sample(range(x.shape[0]), 2)
            idx = list(range(x.shape[0]))
            idx[s_idx[0]] = s_idx[1]
            idx[s_idx[1]] = s_idx[0]
            x = x[idx]
        return x


class RandomTempShift:  # x version
    def __init__(self, max_shift=30, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample):
        p = np.random.rand()
        if isinstance(sample, dict):
            x = sample['x']
        else:
            x = sample
        if p < self.p:
            shift = int(np.clip(np.random.randn() * 0.3, -1, 1) * self.max_shift/5)
            x = np.roll(x, shift, axis=0)

        if isinstance(sample, dict):
            return {'x': x, 'doy': sample['doy']}
        else:
            return x


class RandomTempRemoval:
    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']
        mask = [1 if random.random() < 0.15 else 0 for _ in range(x.shape[0])]
        mask = np.array(mask) == 0

        return {'x': x[mask], 'doy': doy[mask]}


class RandomSampleTimeSteps:
    def __init__(self, sequencelength, rc=False):
        self.sequencelength = sequencelength
        self.rc = rc

    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']

        if self.rc:
            # choose with replacement if sequencelength smaller als choose_t
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # must have x_pad, mask, and doy_pad
            x_pad = x[idxs]
            mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]
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
            else:
                idxs = np.random.choice(x.shape[0], self.sequencelength, replace=False)
                idxs.sort()

                x_pad = x[idxs]
                mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]
        sample['x'] = torch.from_numpy(x_pad).type(torch.FloatTensor),
        sample['doy'] = torch.from_numpy(doy_pad).type(torch.LongTensor)
        sample['mask'] = torch.from_numpy(mask == 0)

        return torch.from_numpy(x_pad).type(torch.FloatTensor), \
               torch.from_numpy(mask == 0), \
               torch.from_numpy(doy_pad).type(torch.LongTensor), torch.tensor([])


def getWeight(x):
    score = np.ones(x.shape[0])
    score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) - 0.2) / 0.6)  # rgb
    cloud = score * 100 > 20
    dark = x[:, [6, 8, 9]].sum(1) < 0.35 # NIR, SWIR1, SWIR2

    ndvi = (x[:, 6] - x[:, 2]) / (x[:, 6] + x[:, 2] + 1e-8)
    ndvi[cloud] = -1
    ndvi[dark] = -1
    ndvi = ndvi.clip(-1, 1)

    weight = np.exp(ndvi)
    weight /= weight.sum()

    return weight