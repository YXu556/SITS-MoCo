import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import moco.loader
import moco.builder
from datasets import *
from models import *


# -------------------------- #
#          dataset           #
# -------------------------- #
def get_sup_dataloader(modelname, datapath, year, batchsize, workers, sequencelength, num, interp, rc,
                       useall=False, nclasses=20, seed=111):
    train_dataaug = RandomTempShift()

    if modelname in ['rf', 'RF']:
        num_train = int(num * 0.9)
        num_val = num - num_train
        traindataset = USCrops(mode='train', root=datapath, year=year, sequencelength=sequencelength,
                               dataaug=train_dataaug, num=num_train, interp=interp, nclasses=nclasses, seed=seed)
        testdataset = USCrops(mode='eval', root=datapath, year=year, sequencelength=sequencelength,
                              useall=useall, num=num_val, nclasses=nclasses, interp=interp, seed=seed)

        X_train = traindataset.X_list
        for i, X in enumerate(X_train):
            X_train[i] = USCrops.transform(traindataset, X, interp=interp, rc=rc)[0].numpy()
        X_train = np.array(X_train).reshape(len(X_train), -1)
        y_train = traindataset.index['classid'].values

        X_test = testdataset.X_list
        for i, X in enumerate(X_test):
            X_test[i] = USCrops.transform(testdataset, X, interp=interp, rc=rc)[0].numpy()
        X_test = np.array(X_test).reshape(len(X_test), -1)
        y_test = testdataset.index['classid'].values

        meta = dict(
            ndims=10*sequencelength,
            num_classes=traindataset.nclasses+1,
        )

        return (X_train, y_train, X_test, y_test), meta

    else:
        num_train = int(num * 0.9)
        num_val = num - num_train
        traindataset = USCrops(mode='train', root=datapath, year=year, sequencelength=sequencelength, dataaug=train_dataaug,
                               useall=useall, num=num_train, randomchoice=rc, interp=interp, nclasses=nclasses, seed=seed)
        valdataset = USCrops(mode='valid', root=datapath, year=year, sequencelength=sequencelength,
                             useall=useall, num=num_val, randomchoice=rc, interp=interp, nclasses=nclasses, seed=seed)
        testdataset = USCrops(mode='eval', root=datapath, year=year, sequencelength=sequencelength,
                              useall=useall, num=num_val, randomchoice=rc, interp=interp, nclasses=nclasses, seed=seed)

        traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=batchsize, shuffle=True,
                                                      num_workers=workers, pin_memory=True)
        valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=batchsize, shuffle=False,
                                                    num_workers=workers)
        testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=batchsize, shuffle=False,
                                                     num_workers=workers, pin_memory=True)
        meta = dict(
            ndims=10,
            num_classes=traindataset.nclasses+1,
        )

        return (traindataloader, valdataloader, testdataloader), meta


def get_moco_dataloader(datapath, year, batchsize, workers, sequencelength, num, rc, seed, useall):
    train_dataaug = transforms.Compose([
        RandomTempShift(),
        RandomAddNoise(),
        RandomTempRemoval(),
        RandomSampleTimeSteps(sequencelength, rc=rc)
    ])

    pretraindataset = MoCoDataset(root=datapath, year=year, sequencelength=sequencelength,
                                  dataaug=train_dataaug, num=num, randomchoice=rc, seed=seed, useall=useall)

    num = len(pretraindataset)
    num_train = int(num * 0.9)
    indices = list(range(num))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:num_train], indices[num_train:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    traindataloader = torch.utils.data.DataLoader(pretraindataset, batch_size=batchsize, sampler=train_sampler,
                                                  num_workers=workers, pin_memory=True, drop_last=True)
    valdataloader = torch.utils.data.DataLoader(pretraindataset, batch_size=batchsize, sampler=valid_sampler,
                                                num_workers=workers, drop_last=True)
    meta = dict(
        ndims=10,
    )

    return traindataloader, valdataloader, meta


def get_bert_dataloader(datapath, year, batchsize, workers, sequencelength, num, rc, seed, useall):
    pretraindataset = BERTDataset(root=datapath, year=year, sequencelength=sequencelength,
                                  num=num, randomchoice=rc, seed=seed, useall=useall)
    num = len(pretraindataset)
    num_train = int(num * 0.9)
    indices = list(range(num))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:num_train], indices[num_train:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    traindataloader = torch.utils.data.DataLoader(pretraindataset, batch_size=batchsize, sampler=train_sampler,
                                                  num_workers=workers, pin_memory=True, drop_last=True)
    valdataloader = torch.utils.data.DataLoader(pretraindataset, batch_size=batchsize, sampler=valid_sampler,
                                                num_workers=workers, drop_last=True)
    meta = dict(
        ndims=10,
    )

    return traindataloader, valdataloader, meta


# -------------------------- #
#           Model            #
# -------------------------- #
def get_model(modelname, ndims, num_classes, sequencelength, device):
    modelname = modelname.lower()  # make case invariant
    if modelname == 'transformer':
        model = TransformerModel(input_dim=ndims, num_classes=num_classes, max_seq_len=sequencelength).to(device)
    elif modelname == 'tempcnn':
        model = TempCNN(input_dim=ndims, num_classes=num_classes, max_seq_len=sequencelength).to(device)
    elif modelname == 'lstm':
        model = LSTM(input_dim=ndims, num_classes=num_classes).to(device)
    elif modelname == 'ltae':
        model = LTAE(input_dim=ndims, num_classes=num_classes, max_seq_len=sequencelength).to(device)
    elif modelname == 'rf':
        model = RandomForestClassifier(n_estimators=500, max_depth=25)
    elif modelname == 'stnet':
        model = STNet(input_dim=ndims, num_classes=num_classes, max_seq_len=sequencelength).to(device)
    else:
        raise ValueError(
            "invalid model argument. choose from 'Transformer', 'TempCNN', 'LSTM', 'LTAE', 'RF', or 'STNet' ")

    return model


def get_moco_model(modelname, device, args):
    modelname = modelname.lower()
    if modelname == 'transformer':
        basemodel = TransformerModel
    elif modelname == 'tempcnn':
        basemodel = TempCNN
    elif modelname == 'lstm':
        basemodel = LSTM
    elif modelname == 'ltae':
        basemodel = LTAE
    elif modelname == 'stnet':
        basemodel = STNet
    else:
        raise ValueError(
            "invalid model - basemodel argument")

    model = moco.builder.MoCo(
        basemodel,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    model.modelname = f'{model.modelname}{basemodel().modelname}'

    model = model.to(device)

    return model


# -------------------------- #
#           Utils            #
# -------------------------- #
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, num_classes=21):
    num = target.shape[0]

    confusion_matrix = get_confusion_matrix(output, target, num_classes)
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(1) - TP
    FN = confusion_matrix.sum(0) - TP

    po = TP.sum() / num
    pe = (confusion_matrix.sum(0) * confusion_matrix.sum(1)).sum() / num ** 2
    if pe == 1:
        kappa = 1
    else:
        kappa = (po - pe) / (1 - pe)

    p = TP / (TP + FP + 1e-12)
    r = TP / (TP + FN + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    oa = po
    kappa = kappa
    macro_f1 = f1.mean()
    weight = confusion_matrix.sum(0) / confusion_matrix.sum()
    weighted_f1 = (weight * f1).sum()
    class_f1 = f1

    return dict(
        oa=oa,
        kappa=kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        class_f1=class_f1,
        confusion_matrix=confusion_matrix
    )


def get_confusion_matrix(y_pred, y_true, num_classes=21):
    idx = y_pred * num_classes + y_true
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def save(model, path="model.pth", **kwargs):
    print(f"saving model to {str(path)}\n")
    model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


def overall_performance(logdir):
    overall_metrics = defaultdict(list)

    for seed in [111, 222, 333, 444, 555]:
        log_dir = Path(logdir.replace(re.findall('Seed\d+', str(logdir))[0], f'Seed{seed}'))
        log_fn = log_dir / f'testlog.csv'
        if log_fn.exists():
            test_metrics = pd.read_csv(log_fn).iloc[0].to_dict()
            for metric, value in test_metrics.items():
                overall_metrics[metric].append(value)

    print(f'Overall result across 5 trials:')
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        if 'loss' in metric or 'f1' in metric or 'kappa' in metric:
            print(f"{metric}: {np.mean(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.2f}")

    print(f'{np.mean(overall_metrics["oa"])*100:.2f}\t'
          f'{np.mean(overall_metrics["kappa"]):.4f}\t'
          f'{np.mean(overall_metrics["weighted_f1"]):.4f}')
    print()

