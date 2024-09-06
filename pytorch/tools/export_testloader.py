# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from models.spike import *

from trainer import *
from tools.test_runner import *
import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.onnx
parser = argparse.ArgumentParser(description='argument setting of network')
# dataset
parser.add_argument('--cuda', default=0, type=int, help='cuda device ID, 0/1')
parser.add_argument('--rep', default=1, type=int,
                    help='repeats for multiple runs')
parser.add_argument('--modeldir', type=str, default='./', help='model directory')
# hyperparameter
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=60,
                    help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_cls', type=float, default=1e-3,
                    help='learning rate for linear classifier')

# dataset
parser.add_argument('--dataset', type=str, default='shar', choices=['oppor', 'ucihar', 'shar', 'hhar'],
                    help='name of dataset')
parser.add_argument('--n_feature', type=int, default=77,
                    help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=30,
                    help='length of sliding window')
parser.add_argument('--n_class', type=int, default=18, help='number of class')
parser.add_argument('--cases', type=str, default='random', choices=['random', 'subject', 'subject_large',
                                                                    'cross_device',
                                                                    'joint_device'], help='name of scenarios')
parser.add_argument('--split_ratio', type=float, default=0.2, help='split ratio of test/val: train(0.64), val(0.16), '
                                                                   'test(0.2)')
parser.add_argument('--target_domain', type=str, default='0', help='the target domain, [0 to 29] for ucihar, '
                                                                   '[1,2,3,5,6,9,11,13,14,15,16,17,19,20,21,'
                                                                   '22,23,24,25,29] for shar, [a-i] for hhar')

# backbone model
parser.add_argument('--backbone', type=str, default='FCN',
                    choices=['FCN', 'RESNET', 'DCL', 'SDCL', 'SFCN', 'SRESNET', 'CASNN', 'CASRESNET'], help='name of framework')

# log
parser.add_argument('--logdir', type=str, default='log/', help='log directory')

# hhar
parser.add_argument('--device', type=str, default='Phones', choices=['Phones', 'Watch'],
                    help='data of which device to use (random case);'
                         ' data of which device to be used as training data (cross-device case,'
                         ' data from the other device as test data)')

# spike
parser.add_argument('--tau', type=float, default=0.5, help='decay for LIF')
parser.add_argument('--thresh', type=float, default=1.0,
                    help='threshold for LIF')
parser.add_argument('--eval', action='store_true', help='Evaluation model')
parser.add_argument('--eescan', action='store_true',
                    help='Early exit threshold scanning mode')
parser.add_argument('--use_random', action='store_true', help='For the case that use random settings to train but use leave one domain settings to test.')



# Example: python export_testloader.py --dataset ucihar --backbone FCN --cases subject_large --target_domain 1
if __name__ == '__main__':
    args = parser.parse_args()

    train_loaders, val_loader, test_loader = setup_dataloaders(args)

    samples = []
    targets = []
    for sample, target, domain in iter(test_loader):
        sample_ = sample.transpose(1, 2)
        sample_ = sample_.numpy().flatten()        
        target_ = target.numpy().flatten()
        samples.append(sample_)
        targets.append(target_)
    samples = np.array(samples)
    targets = np.array(targets)
    # print(f'./testloader/samples_{args.dataset}_{args.target_domain if args.cases == "subject_large" else "_"}.npy') rgets)
    
    np.savetxt(f'./testloader/samples_{args.dataset}_{args.target_domain if (args.cases == "subject_large" or args.cases == "subject") else "_"}.txt', samples, fmt='%.4f', delimiter='\n')
    np.savetxt(f'./testloader/targets_{args.dataset}_{args.target_domain if (args.cases == "subject_large" or args.cases == "subject") else "_"}.txt', targets, fmt='%d', delimiter='\n')

        