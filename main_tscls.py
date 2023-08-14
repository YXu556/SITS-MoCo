"""
This script is for time series classification task.
"""
import copy
import argparse
from tqdm import tqdm
from joblib import dump, load

import torch.optim
import torch.nn.functional as F

from utils import *

DATAPATH = Path(r"data/US-toy")  # todo replace your datapath here
YEARS = [2019]#, 2020, 2021]  # 3 testing years
SEEDS = [111]#, 222, 333, 444, 555]  # 5 repeated trails


def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models.')
    parser.add_argument('model', type=str, default="STNet",
                        help='select model architecture.')
    parser.add_argument('--use-doy', action='store_true',
                        help='whether to use doy pe with trsf')
    parser.add_argument('--rc', action='store_true',
                        help='whether to random choice the time series data')
    parser.add_argument('--interp', action='store_true',
                        help='whether to interplate the time series data')
    parser.add_argument('--useall', action='store_true',
                        help='whether to use all data for training')
    parser.add_argument('-n', '--num', default=3000, type=int,
                        help='number of labeled samples (training and validation) (default 3000)')
    parser.add_argument('-c', '--nclasses', type=int, default=20,
                        help='num of classes (default: 20)')
    parser.add_argument('--year', type=int, default=2019,
                        help='year of dataset')
    parser.add_argument('-seq', '--sequencelength', type=int, default=70,
                        help='Maximum length of time series data (default 70)')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--schedule', default=None, nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('-l', '--logdir', type=str, default="./results",
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--freeze', action='store_true',
                        help='freeze pretrain model')
    args = parser.parse_args()

    args.dataset = 'USCrops'
    args.datapath = DATAPATH

    modelname = args.model.lower()
    if args.interp and modelname in ['rf', 'tempcnn', 'lstm']:
        args.interp = True
    else:
        args.interp = False

    if args.interp:
        args.rc_str = 'Int'
    elif args.rc:
        args.rc_str = 'RC'
    else:
        args.rc_str = 'Pad'

    if args.use_doy:
        if args.suffix:
            args.suffix = 'doy_' + args.suffix
        else:
            args.suffix = 'doy'

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def train(args):
    print("=> creating dataloader")
    data, meta = get_sup_dataloader(args.model, args.datapath, args.year, args.batchsize, args.workers,
                                    args.sequencelength,
                                    args.num, args.interp, args.rc, args.useall, args.nclasses, args.seed)

    num_classes = meta["num_classes"]
    ndims = meta["ndims"]

    if args.model in ['rf', 'RF']:
        X_train, y_train, X_test, y_test = data
    else:
        traindataloader, valdataloader, testdataloader = data

    print("=> creating model '{}'".format(args.model))
    device = torch.device(args.device)
    model = get_model(args.model, ndims, num_classes, args.sequencelength, device)

    if args.model in ['RF', 'rf']:
        if args.suffix:
            logdir = Path(args.logdir) / (f'T_RF_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}_{args.suffix}')
        else:
            logdir = Path(args.logdir) / (f'T_RF_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}')
        logdir.mkdir(exist_ok=True, parents=True)
        best_model_path = logdir / 'model_best.joblib'
        if not args.eval:
            print('training Random Forest...')
            model.fit(X_train, y_train)
            print(f"saving model to {str(best_model_path)}\n")
            dump(model, best_model_path)

        print('Restoring best model weights for testing...')
        model = load(best_model_path)

        y_pred = model.predict(X_test)
        scores = accuracy(y_test, y_pred, args.nclasses + 1)

        scores_msg = ", ".join(
            [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results : \n\n {scores_msg} \n\n")

        scores['epoch'] = 'test'
        conf_mat = scores.pop('confusion_matrix')
        class_f1 = scores.pop('class_f1')

        log_df = pd.DataFrame([scores]).set_index("epoch")
        log_df.to_csv(logdir / f"testlog.csv")
        np.save(logdir / f"test_conf_mat.npy", conf_mat)
        np.save(logdir / f"test_class_f1.npy", class_f1)

        return logdir

    print(f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}")
    model.apply(weight_init)
    finetune = False
    if args.pretrained is not None:
        finetune = True
        path = Path(args.pretrained).absolute().relative_to(Path(__file__).absolute().parent)
        print("=> loading checkpoint '{}'".format(str(path)))
        pretrain_model = torch.load(path)['model_state']
        model_dict = model.state_dict()
        if 'moco' in str(path.parts[-2]).lower():
            state_dict = {}
            for k in list(pretrain_model.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q') and not k.startswith('encoder_q.decoder') and not k.startswith(
                        'encoder_q.classification') and not k.startswith('encoder_q.position_enc.pe'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = pretrain_model[k]  # module.
        else:
            state_dict = {k: v for k, v in pretrain_model.items() if
                          k in model_dict.keys() and 'decoder' not in k and 'position_enc.pe' not in k}

        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if args.freeze:
        for name, param in model.named_parameters():
            if not name.startswith('decoder'):
                param.requires_grad = False

    if finetune:
        model.modelname = f'F_{path.parts[-2].split("_")[1][:2]}_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}'
    elif args.useall:
        model.modelname = f'T_{model.modelname}_{args.rc_str}_{args.year}'
    else:
        model.modelname = f'T_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}'

    if args.suffix:
        model.modelname += f'_{args.suffix}'

    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / 'model_best.pth'
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    if not args.eval:
        log = list()
        val_loss_min = np.Inf
        print(f"Training {model.modelname}")
        for epoch in range(args.epochs):
            if args.warmup_epochs > 0:
                if epoch == 0:
                    lr = args.learning_rate * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                elif epoch == args.warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.learning_rate

            if args.schedule is not None:
                adjust_learning_rate(optimizer, epoch, args)
            train_loss = train_epoch(model, optimizer, criterion, traindataloader, device, args)
            val_loss, scores = test_epoch(model, criterion, valdataloader, device, args)
            scores_msg = ", ".join(
                [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
            print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

            if val_loss < val_loss_min:
                not_improved_count = 0
                save(model, path=best_model_path, criterion=criterion)
                val_loss_min = val_loss
                print(f'lowest val loss in epoch {epoch + 1}\n')
            else:
                not_improved_count += 1

            scores["epoch"] = epoch + 1
            scores["trainloss"] = train_loss
            scores["testloss"] = val_loss
            log.append(scores)

            log_df = pd.DataFrame(log).set_index("epoch")
            log_df.to_csv(Path(logdir) / "trainlog.csv")

            if not_improved_count >= 10:
                print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
                break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")

    # test
    print('Restoring best model weights for testing...')
    checkpoint = torch.load(best_model_path)
    state_dict = {k: v for k, v in checkpoint['model_state'].items()}
    criterion = checkpoint['criterion']
    torch.save({'model_state': state_dict, 'criterion': criterion}, best_model_path)
    model.load_state_dict(state_dict)

    test_loss, scores = test_epoch(model, criterion, testdataloader, device, args)
    scores_msg = ", ".join(
        [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
    print(f"Test results: \n\n {scores_msg}")

    scores['epoch'] = 'test'
    scores['testloss'] = test_loss
    conf_mat = scores.pop('confusion_matrix')
    class_f1 = scores.pop('class_f1')

    log_df = pd.DataFrame([scores]).set_index("epoch")
    log_df.to_csv(logdir / f"testlog.csv")
    np.save(logdir / f"test_conf_mat.npy", conf_mat)
    np.save(logdir / f"test_class_f1.npy", class_f1)

    return logdir


def train_epoch(model, optimizer, criterion, dataloader, device, args):
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (X, y) in iterator:
            X = recursive_todevice(X, device)
            y = y.to(device)

            optimizer.zero_grad()
            if args.use_doy:
                logits = model(X, use_doy=True)
            else:
                logits = model(X)
            out = F.log_softmax(logits, dim=-1)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")

            losses.update(loss.item(), X[0].size(0))

    return losses.avg


def test_epoch(model, criterion, dataloader, device, args):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                X = recursive_todevice(X, device)
                y = y.to(device)

                if args.use_doy:
                    logits = model(X, use_doy=True)
                else:
                    logits = model(X)
                out = F.log_softmax(logits, dim=-1)

                loss = criterion(out, y)
                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), X[0].size(0))

                y_true_list.append(y)
                y_pred_list.append(out.argmax(-1))
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()

    scores = accuracy(y_true, y_pred, args.nclasses + 1)

    return losses.avg, scores


def main():
    args = parse_args()
    years = YEARS
    for year in years:
        print(f' ===================== {year} ======================= ')
        args.year = year
        seeds = SEEDS
        print('seed in', seeds)
        for seed in seeds:
            args.seed = seed
            print(f'Seed = {args.seed} --------------- ')

            SEED = args.seed
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True

            logdir = train(args)
        overall_performance(str(logdir))


if __name__ == '__main__':
    main()
