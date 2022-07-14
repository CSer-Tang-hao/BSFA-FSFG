from __future__ import print_function
from __future__ import division

import sys
import time
import argparse
import os.path as osp
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


from tqdm import tqdm
from models.net import Model

from data_manager import DataManager
from utils import AverageMeter
from utils import Logger
from utils.torchtools import one_hot

parser = argparse.ArgumentParser(description='Test image model with 5-way classification')

# Datasets
parser.add_argument('--dataset', type=str, default='CUB-200-2011')
parser.add_argument('--model', default='R', type=str,
                    help='C for ConvNet, R for ResNet')
parser.add_argument('--workers', default= 8, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=84,
                    help="height of an image (default: 84)")
parser.add_argument('--width', type=int, default=84,
                    help="width of an image (default: 84)")
# Optimization options
parser.add_argument('--train-batch', default=4, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=8, type=int,
                    help="test batch size")
# Architecture
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--save-dir', type=str, default='./result')
# parser.add_argument('--resume', type=str, default='./result/best_model.pth.tar')
# FewShot settting
parser.add_argument('--nKnovel', type=int, default=5,
                    help='number of novel categories')
parser.add_argument('--nExemplars', type=int, default=1,
                    help='number of training examples per novel category.')
parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                    help='number of test examples for all the novel category when training')
parser.add_argument('--train_epoch_size', type=int, default=1200,
                    help='number of episodes per epoch when training')
parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                    help='number of test examples for all the novel category')
parser.add_argument('--epoch_size', type=int, default=2000,
                    help='number of batches per epoch')
# Miscs
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-devices', default='0', type=str)
parser.add_argument('--weight_global', type=float, default=1.)
parser.add_argument('--weight_local', type=float, default=0.1)
args = parser.parse_args()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    args.save_dir = osp.join(args.save_dir, args.dataset, args.model, 'global_' + str(args.weight_global) + '_local_' + str(args.weight_local))
    args.resume = osp.join(args.save_dir, 'best_model.pth.tar')
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    model = Model(num_classes=args.num_classes, backbone=args.model)
    # load the model
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint from '{}'".format(args.resume))
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{}]  {}'.format(tm,'Testing'))

    if use_gpu:
        model = model.cuda()

    test(model, testloader, use_gpu)



def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(testloader)):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)


            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            s1 = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            s1 = s1.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

           
            _, preds = torch.max(s1.detach().cpu(), 1)

            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() 
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{}] Accuracy: {:.2%}, std: {:.2%}'.format(tm, accuracy, ci95))


    return accuracy


if __name__ == '__main__':
    main()
