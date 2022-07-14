from __future__ import print_function
from __future__ import division

import os
import sys
import datetime
import time

import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from models.net import Model

from data_manager import DataManager
from utils.losses import CrossEntropyLoss
from utils.optimizers import init_optimizer

from utils.iotools import save_checkpoint
from utils import AverageMeter
from utils import Logger
from utils.torchtools import one_hot, adjust_learning_rate



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = False  #
    cudnn.deterministic = True

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.model, 'global_' + str(args.weight_global) + '_local_' + str(args.weight_local))

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    Dataset = DataManager(args, use_gpu)
    trainloader, testloader = Dataset.return_dataloaders()
    print('Initializing image data manager')


    model = Model(num_classes=args.num_classes, backbone=args.model) ###num_classer=100
    if use_gpu:
        model = model.cuda()
    print("save checkpoint to '{}'".format(args.save_dir))


    criterion1 = CrossEntropyLoss().cuda()
    criterion2 = torch.nn.CrossEntropyLoss().cuda()
    optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)


    if args.model == 'C':
        args.max_epoch = 160
        args.LUT_lr = [(100, 0.1), (170, 0.006), (240, 0.0012), (300, 0.00024)]
    elif args.model == 'R':
        args.max_epoch = 90
        args.LUT_lr = [(60, 0.1), (70, 0.006), (80, 0.0012),(90,0.00024)]

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        train(args, epoch, model, criterion1, criterion2, optimizer, trainloader, learning_rate, use_gpu)
        train_time += round(time.time() - start_train_time)

        if epoch == 0 or epoch % 5 == 0 or epoch >= (args.LUT_lr[0][0] - 1):

            acc = test(model, testloader, use_gpu)
            is_best = acc > best_acc
            
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(args, epoch, model, criterion1, criterion2, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids_1, pids_2) in enumerate(tqdm(trainloader)):

        data_time.update(time.time() - end)
        
        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            images_test, labels_test = images_test.cuda(), labels_test.cuda()
            pids_1 = pids_1.cuda()
            pids_2 = pids_2.cuda()

        pids = torch.cat((pids_1,pids_2),dim=1)
        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()


        s1, s2, glo1, glo2= model(images_train, images_test, labels_train_1hot, labels_test_1hot)

        loss_global1 = criterion2(glo1,pids.view(-1))
        loss_global2 = criterion2(glo2,pids.view(-1))
        loss_global  = 0.5 * loss_global1 + 0.5 * loss_global2



        loss_xcos1 = criterion1(s1, labels_test.view(-1))
        loss_xcos2 = criterion1(s2, labels_test.view(-1))
        loss_xcos = 0.5*loss_xcos1 + 0.5*loss_xcos2

        loss = loss_global * args.weight_global + loss_xcos * args.weight_local

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids_2.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{0}] '
          'Epoch{1} '
          'lr: {2} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(tm,
           epoch+1, learning_rate, batch_time=batch_time, 
           data_time=data_time, loss=losses))


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(testloader)):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()


            q_batch_size,num_test_examples = images_test.size(0), images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            s3 = model(images_train, images_test, labels_train_1hot, labels_test_1hot)

            s3 = s3.view(q_batch_size * num_test_examples, -1)
            labels_test = labels_test.view(q_batch_size * num_test_examples)


            _, preds = torch.max(s3.detach().cpu(), 1)
            # print(preds)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(q_batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (q_batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)  ### 2000
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{}] Accuracy: {:.2%}, std: {:.2%}'.format(tm,accuracy, ci95))

    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--dataset', type=str, default='CUB-200-2011')
    parser.add_argument('--model', default='R', type=str,
                        help='C for ConvNet, R for ResNet')
    parser.add_argument('--workers', default=8, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--train-batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=4, type=int,
                        help="test batch size")
    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--num_classes', type=int, default=100)
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./result/')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0', type=str)
    # ************************************************************
    # Loss settings
    # ************************************************************
    parser.add_argument('--weight_global', type=float, default=1.)
    parser.add_argument('--weight_local', type=float, default=0.1)
    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')
    parser.add_argument('--train_nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='val', type=str,
                        help='use test or val dataset to early stop')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    main(args)
