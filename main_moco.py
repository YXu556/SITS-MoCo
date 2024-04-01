"""
This script is for SITS-MoCo pre-training task.
"""
import argparse
import torch.nn as nn
from tqdm import tqdm

from utils import *


DATAPATH = Path(r"data/US-toy")  # todo replace your datapath here


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-Train a time series feature extractor.')
    parser.add_argument('model', type=str, default="transformer",
                        help='select pretrain method model architecture (default Transformer).')
    parser.add_argument('--use-doy', action='store_true',
                        help='whether to use doy pe with trsf')
    parser.add_argument('--rc', action='store_true',
                        help='whether to random choice the time series data')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--useall', action='store_true',
                        help='whether to use all data for training (upper bound)')
    parser.add_argument('-n', '--num', default=3000, type=int,
                        help='number of labeled samples (training and validation) (default 3000)')
    parser.add_argument('-c', '--nclasses', type=int, default=20,
                        help='num of classes (default: 20)')
    parser.add_argument('--sequencelength', type=int, default=70,
                        help='Maximum length of time series data (default 70)')
    parser.add_argument('--year', type=int, default=2019,
                        help='year of dataset')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='warmup epochs')  # todo
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--schedule', default=None, nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 0)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=111, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('-l', '--logdir', type=str, default="./results",
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    args = parser.parse_args()

    args.datapath = DATAPATH

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rc:
        args.rc_str = 'RC'
    else:
        args.rc_str = 'Pad'

    if args.use_doy:
        if args.suffix:
            args.suffix = 'doy_' + args.suffix
        else:
            args.suffix = 'doy'

    if args.seed is not None:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    return args


def train(args):
    print("=> creating dataloader")
    traindataloader, valdataloader, meta = get_moco_dataloader( args.datapath, args.year, args.batchsize, args.workers,
                                                                args.sequencelength, args.num, args.rc, args.seed, args.useall)

    print("=> creating model")
    device = torch.device(args.device)
    model = get_moco_model(args.model, device, args)

    if args.useall:
        model.modelname = f'P_{model.modelname}_{args.rc_str}_{args.year}'
    else:
        model.modelname = f'P_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}'

    if args.suffix:
        model.modelname += f'_{args.suffix}'

    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / 'model_best.pth'
    print(f"Logging results to {logdir}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    log = list()
    val_loss_min = np.Inf
    if args.resume:
        path = Path(args.resume).absolute().relative_to(Path(__file__).absolute().parent)
        print("=> loading checkpoint '{}'".format(str(path)))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        args.start_epoch = torch.load(path)['epoch'] + 1
        val_loss_min = checkpoint['val_loss_min']
        not_improved_count = checkpoint['not_improved_count']

        print("=> loaded checkpoint '{}'".format(str(path)))

        log_fn = path.parent / "trainlog.csv"
        log = pd.read_csv(log_fn).to_dict('records')[:args.start_epoch]

    print(f"Pre-training {args.model}...")
    for epoch in range(args.start_epoch, args.epochs):

        if args.warmup_epochs > 0:
            if epoch == 0:
                lr = args.learning_rate * 0.01
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif epoch == args.warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate
        if args.schedule is not None:
            adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(model, optimizer, criterion, traindataloader, device, use_doy=args.use_doy)
        val_loss = test_epoch(model, criterion, valdataloader, device, use_doy=args.use_doy)

        print(f"epoch {epoch}: trainloss {train_loss:.4f}, valloss {val_loss:.4f} ")

        scores = {}
        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = val_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(logdir / "trainlog.csv")

        if val_loss < val_loss_min:
            not_improved_count = 0
            val_loss_min = val_loss
            save(model, path=best_model_path, epoch=epoch, val_loss_min=val_loss, not_improved_count=not_improved_count)
        else:
            not_improved_count += 1

        if not_improved_count >= 10:
            print("\nValidation performance didn\'t improve for 20 epochs. Training stops.")
            break


def train_epoch(model, optimizer, criterion, dataloader, device, use_doy=False):
    losses = AverageMeter('Loss', ':.4e')
    model.train()

    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (data_q, data_k) in iterator:
            data_q = recursive_todevice(data_q, device)
            data_k = recursive_todevice(data_k, device)

            # compute output
            output, target = model(data_q=data_q, data_k=data_k, use_doy=use_doy)
            loss = criterion(output, target)

            # compufte gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iterator.set_description(f"train loss={loss:.2f}")

            losses.update(loss.item(), data_q[0].size(0))

    return losses.avg


def test_epoch(model, criterion, dataloader, device, use_doy=False):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (data_q, data_k) in iterator:
                data_q = recursive_todevice(data_q, device)
                data_k = recursive_todevice(data_k, device)

                output, target = model(data_q=data_q, data_k=data_k, use_doy=use_doy)
                loss = criterion(output, target)

                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), data_q[0].size(0))

    return losses.avg



def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
