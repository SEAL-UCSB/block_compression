import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sparselayer import BlocksparseConv, BlocksparseLinear
from sparsemodel import BlocksparseModel
from config import configuration

model_names = configuration.keys()

parser = argparse.ArgumentParser(description='Fine-tuning for deep block compression')
parser.add_argument('data', metavar='DIR', help='Path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
parser.add_argument('--batch-size', '-bs', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true', help='only evaluate model on validation set')

args = parser.parse_args()
best_prec1 = 0

def main():
    print("=> load pre-trained model %s" % args.arch)
    model = models.__dict__[args.arch](pretrained=True)
    print("=> create sparse model")
    model = BlocksparseModel(model, configuration[args.arch].block_sizes, configuration[args.arch].pruning_rates)
    model = torch.nn.DataParallel(model).cuda()
 
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optional resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint %s" % args.resume)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint %s (epoch %d)" % (args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at %s" % args.resume)

    cudnn.benchmark = True

    # load data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # optional evaluate only
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust learning rate
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)

        # save best and checkpoint
        checkpoint_fn = 'checkpoint.pth.tar'
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()},
            checkpoint_fn)
        if is_best:
            shutil.copyfile(checkpoint_fn, 'model_best.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        y = y.cuda(non_block=True)
        y_ = model(x)

        loss = criterion(y_, y)
        prec1, prec5 = accuracy(y_, y, topk=(1, 5))
        
        losses.update(loss.item(), x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time

        if i % args.print_freq == 0:
            print(
                "Epoch: [%d][%d/%d]\t" % (epoch, i, len(train_loader)) + \
                "Time %0.3f (%0.3f)\t" % (batch_time.val, batch_time.avg) + \
                "Data %0.3f (%0.3f)\t" % (data_time.val, data_time.avg) + \
                "Loss %0.4f (%0.4f)\t" % (losses.val, losses.avg) + \
                "Prec@1 %0.3f (%0.3f)\t" % (top1.val, top1.avg) + \
                "Prec@5 %0.3f (%0.3f)\t" % (top5.val, top5.avg))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, y) in enumerate(val_loader):
            y = y.cuda(non_blocking=True)
            y_ = model(x)

            loss = criterion(y_, y)
            prec1, prec5 = accuracy(y_, y, topk=(1, 5))
            
            losses.update(loss.item(), x.size(0))
            top1.update(prec1[0], x.size(0))
            top5.update(prec5[0], x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Test: [%d/%d]\t" % (i, len(val_loader)) + \
                    "Time %0.3f (%0.3f)\t" % (batch_time.val, batch_time.avg) + \
                    "Loss %0.4f (%0.4f)\t" % (losses.val, losses.avg) + \
                    "Prec@1 %0.3f (%0.3f)\t" % (top1.val, top1.avg) + \
                    "Prec@5 %0.3f (%0.3f)\t" % (top5.val, top5.avg))

        print(" * Prec@1 %0.3f Prec@5 %0.3f" % (top1.avg, top5.avg))
    return top1.avg


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, vale, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()