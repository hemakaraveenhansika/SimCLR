import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data import ContrastiveBatchSampler
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./data/images',
                    help='path to dataset')
parser.add_argument('-record_dir', metavar='record_dir', default='./', help='path to record_dir')

parser.add_argument('-dataset-name', default='contrastive',
                    help='dataset name', choices=['stl10', 'cifar10','contrastive'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--train-image-list', default="./data/train_list.txt", type=str, help='Image name list path')
parser.add_argument('--val-image-list', default="./data/val_list.txt", type=str, help='Validation Image name list path')
parser.add_argument('--train-image-limit', default=42405, type=int, help='Train images limit')
parser.add_argument('--val-image-limit', default=6079, type=int, help='Val images limit')
parser.add_argument('--resume', metavar='resume', default='/content/SimCLR/currrent_checkpoint.pth.tar', help='path to resume model')
parser.add_argument('--result_dir', metavar='RESULT_DIR', default='./', help='path to result dir')
parser.add_argument('--arch-weights',default='./chexnet.pth.tar',type=str, help='path to arch weights')

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views,args.train_image_list)
    val_dataset = dataset.get_dataset(args.dataset_name, args.n_views,args.val_image_list)
    train_sampler = ContrastiveBatchSampler(args.batch_size,args.train_image_limit,dataset=train_dataset,drop_last=False,seed=int(args.seed))
    val_sampler = ContrastiveBatchSampler(args.batch_size,args.val_image_limit,dataset=val_dataset,drop_last=False,seed=int(args.seed))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.workers, pin_memory=True, batch_sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=args.workers, pin_memory=True, batch_sampler=val_sampler)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim,arch_weights=args.arch_weights)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader,val_loader)


if __name__ == "__main__":
    main()
