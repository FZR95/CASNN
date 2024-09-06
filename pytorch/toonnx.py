# encoding=utf-8
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from models.spike import *

from trainer import *
from tools.test_runner import *
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import numpy as np
import os
from copy import deepcopy
import fitlog
from utils import tsne, mds, _logger, hook_layers
from ptflops import get_model_complexity_info
import torch.onnx
from data_preprocess.base_loader import base_loader
from torch.utils.data import DataLoader
from colorama import init, Fore, Back, Style
init(autoreset=True)
parser = argparse.ArgumentParser(description='argument setting of network')
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
parser.add_argument('--casnn_suffix', type=str, default='', help='#1 or #n')
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
parser.add_argument('--tee', type=int, nargs='+', help='Early Exit Threshold for CASNN')

# exampe: python toonnx.py --dataset ucihar --backbone SFCN --lr 1e-3 --batch_size 128 --cases random --tau 0.75 --thresh 0.5
if __name__ == '__main__':
    args = parser.parse_args()
    #* Set device
    DEVICE = torch.device('cpu')
    print(Fore.GREEN + 'Using device:' + str(DEVICE) + ' ,dataset:' + args.dataset)

    #* Prepare model name to load or create
    args.model_name = args.modeldir + '/'

    if args.backbone == 'CASNN':
        args.model_name += 'SFCN'
    else:
        args.model_name += args.backbone 

    args.model_name += '_' + args.dataset + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_sw' + str(args.len_sw) + '_lod' + str(args.target_domain)
    
    if args.backbone == 'CASNN' or args.backbone == 'SFCN':
        args.model_name += '_tau' + str(args.tau) + '_thresh' + str(args.thresh)
        
    print(Fore.BLUE + 'Using Model: ' +  args.model_name + '.pt')

    #* Load Testset
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
    if args.dataset == 'shar':
        args.n_feature = 3
        args.len_sw = 151
        args.n_class = 17
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
    print(Fore.CYAN + 'Using exist testloader files: ' + 'testloader/' + 'samples_' + args.dataset + '_' + args.target_domain + '.txt')
    samples = np.loadtxt('testloader/' + 'samples_' + args.dataset + '_' + args.target_domain + '.txt').reshape(-1, args.n_feature, args.len_sw)
    samples = samples.transpose(0, 2, 1)
    targets = np.loadtxt('testloader/' + 'targets_' + args.dataset + '_' + args.target_domain + '.txt')
    domains = np.full(targets.shape, args.target_domain)
    dataset = base_loader(samples, targets, domains)
    test_loader = DataLoader(dataset)

    snn_params = {"tau": args.tau, "thresh": args.thresh}

    save_dir = 'results/'
    best_model = torch.load(save_dir + args.model_name + '.pt', map_location=DEVICE)['model_state_dict']

    if args.backbone == 'FCN':
        model_test = FCN_ONNX(n_channels=args.n_feature,
                         n_classes=args.n_class, backbone=False)
    elif args.backbone == 'SFCN':
        model_test = SFCN_ONNX(n_channels=args.n_feature,
                          n_classes=args.n_class, backbone=False, **snn_params)
    elif args.backbone == 'CASNN':
        model_test = CASNN_ONNX(n_channels=args.n_feature, n_classes=args.n_class,
                           backbone=False, ee_thresh=args.tee,  **snn_params)
    else:
        raise NotImplementedError
    
    ipt = None
    for sample, target, domain in iter(test_loader):
        sample = sample.transpose(1, 2)
        ipt = sample
        break
    
    ipt = torch.ones(ipt.shape)

    model_test.load_state_dict(best_model, strict=False)
    model_test = model_test.to(DEVICE)

    model_name = args.model_name+'.onnx'
    if args.backbone == 'CASNN':
        model_name = 'CASNN' + '#' + args.casnn_suffix + args.model_name.split('SFCN')[-1] + '.onnx'
    else:
        model_name = args.model_name.split('/')[-1] + '.onnx'


    torch.onnx.export(model_test, ipt, 
                    'results/onnx/hhar_lod/' + model_name,                   
                    verbose=False,
                    export_params=True,
                    input_names=['input'],
                    output_names=['output'],
                    custom_opsets={"custom":1},
                    dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})

    # model_test.eval()
    # modules2fuse = [
    #     ['conv_block1.0', 'conv_block1.1', 'conv_block1.2']
    # ]
    # fused = torch.ao.quantization.fuse_modules(model_test, modules2fuse)
    # print(fused)
    # torch.onnx.export(fused, ipt, 'results/onnx/' + 'f_' + model_name,                      
    #                 verbose=False,
    #                 input_names=['input'],
    #                 output_names=['output'],
    #                 custom_opsets={"custom":1})
    # print(model_test)