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
import torchsummary

from data_preprocess.base_loader import base_loader
from torch.utils.data import DataLoader
from colorama import init, Fore, Back, Style
init(autoreset=True)
# fitlog.debug()

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

# create directory for saving and plots
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def train(args, train_loaders, val_loader, model, DEVICE, optimizer, criterion):
    min_val_loss = 0

    acc_epoch_list = []
    val_acc_epoch_list = []
    for epoch in range(args.n_epoch):
        if epoch % 5 == 0:
            print(f'\nEpoch : {epoch}')

        train_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        model.train()
        for loader_idx, train_loader in enumerate(train_loaders):
            for idx, (sample, target, domain) in enumerate(train_loader):
                n_batches += 1
                sample, target = sample.to(
                    DEVICE).float(), target.to(DEVICE).long()
                sample = sample.transpose(1, 2)
                out, _ = model(sample)
                loss = criterion(out, target)
                train_loss = train_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(out.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
        acc_train = float(correct) * 100.0 / total
        # fitlog.add_loss(train_loss / n_batches, name="Train Loss", step=epoch)
        # fitlog.add_metric({"dev": {"Train Acc": acc_train}}, step=epoch)
        acc_epoch_list += [round(acc_train, 2)]
        if epoch % 5 == 0:
            print(f'Train Loss     : {train_loss / n_batches:.4f}\t | \tTrain Accuracy     : {acc_train:2.4f}\n')

        if val_loader is None:
            best_model = deepcopy(model.state_dict())
            model_dir = save_dir + args.model_name + '.pt'
            if epoch % 5 == 0:
                print('Saving models to {}'.format(model_dir))
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       model_dir)
        else:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                n_batches = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    n_batches += 1
                    sample, target = sample.to(
                        DEVICE).float(), target.to(DEVICE).long()
                    sample = sample.transpose(1, 2)
                    out, _ = model(sample)
                    loss = criterion(out, target)
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                # fitlog.add_loss(val_loss / n_batches,
                #                 name="Val Loss", step=epoch)
                # fitlog.add_metric({"dev": {"Val Acc": acc_val}}, step=epoch)
                print(f'Val Loss     : {val_loss / n_batches:.4f}\t | \tVal Accuracy     : {acc_val:2.4f}\n')

                val_acc_epoch_list += [round(acc_val, 2)]

                if acc_val >= min_val_loss:
                    min_val_loss = acc_val
                    best_model = deepcopy(model.state_dict())
                    print('update')
                    model_dir = save_dir + args.model_name + '.pt'
                    print('Saving models to {}'.format(model_dir))
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                               model_dir)

    return best_model


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def flops_info(model, data_loader):
    print('\n\nFlops Info: ')
    sample = next(iter(data_loader))[0].transpose(1, 2)
    shape = tuple(sample.size()[1:])
    print('input shape: ', shape)
    torchsummary.summary(model, shape)
    macs, params = get_model_complexity_info(
        model, shape, as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def tee_scan(args, test_loader, best_model, DEVICE, criterion, description):
    ee_logger = []    
    tee_list = []
    suffix = ""
    layer_num = 3  
    file_save_path = f'eescanlog/{args.dataset}_{args.target_domain}{suffix}_test.pkl'
    
    ScanMode = "Coarse"
    if ScanMode=="Single":
        l = 3 # 1,2,3
        suffix = f"_l{l}"
        t_opts = [
            np.arange(228, 292, 8),
            np.arange(144, 176, 8),
            np.arange(80, 120, 8)]
        tee = [248,156,96]
        for i in t_opts[l-1]:
            tee[l-1] = i
            tee_list.append(tee.copy())
    if ScanMode=="Grid":
        suffix = "_g"
        t_opts = [
            np.arange(228, 292, 8),
            np.arange(144, 176, 8),
            np.arange(114, 146, 8)]
        for i in t_opts[0]:
            for j in t_opts[1]:
                for k in t_opts[2]:
                    tee_list.append([i,j,k])
    if ScanMode=="Coarse":
        suffix = ""
        t_opts = [
            np.arange(0, 500, 8),
            np.arange(0, 500, 8),
            np.arange(0, 500, 8)]        
        for i in range(layer_num):  
            for t in t_opts[i]:
                tee = [0, 0, 0]
                tee[i] = t
                tee_list.append(tee)
    
    print(f"\nScan mode: {ScanMode}, will save as: {file_save_path}")
    print(f"{len(tee_list)} thresholds will be tested: \n")
    total_time = 0
    
    for i, tee in enumerate(tee_list):
        t0 = time.time()
        model_test = CASNN(n_channels=args.n_feature,
                           n_classes=args.n_class, backbone=False, ee_thresh=tee, **snn_params)
        model_test.load_state_dict(best_model, strict=False)
        model_test = model_test.to(DEVICE)
        test_ee(args, test_loader, model_test, DEVICE, criterion,
                save_log=False, ee_logger=ee_logger)
        t1 = time.time()
        time_per_tee = t1 - t0
        print(f"Time per tee: {time_per_tee:.2f} sec | Estimated remain {(len(tee_list)-i)*time_per_tee:.1f} sec.")
        total_time += time_per_tee
    print(f'Scan finished with {total_time:.1f} sec.')
    import pickle
    with open(file_save_path, 'wb') as file:
        pickle.dump([tee_list, ee_logger], file)
    print(f'Scanlog saved as: ' + file_save_path)
    exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    #* Set device
    DEVICE = torch.device('cuda:' + str(args.cuda)
                          if torch.cuda.is_available() else 'cpu')
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
        
    # args.model_name = 'paper/SFCN_ucihar_lr0.0001_bs128_sw30_lod1'
    print(Fore.BLUE + 'Using Model: ' +  args.model_name + '.pt')
    if args.backbone == 'CASNN':
        print(Fore.BLUE + 'with early exit threshold: ' + str(args.tee))

    #* Enable Early Exit scanning
    if args.eescan:
        args.eval = True

    #* Testing one by one to simulate the real world scenario
    if args.eval:
        args.batch_size = 1

    #* Params for SNNs
    snn_params = {"tau": args.tau, "thresh": args.thresh}

    acc_list = []
    model_test = None
    test_loader = None
    training_start = datetime.now()
    for r in range(args.rep):
        # fix random seed for reproduction
        seed_all(seed=1000 + r)
        
        test_loader = None
        # Training
        if not args.eval:
            train_loaders, val_loader, test_loader = setup_dataloaders(args)
            
            if args.backbone == 'FCN':
                model = FCN(n_channels=args.n_feature,
                            n_classes=args.n_class, backbone=False)
            elif args.backbone == 'RESNET':
                model = RESNET(n_channels=args.n_feature,
                             n_classes=args.n_class, backbone=False)
            elif args.backbone == 'DCL':
                model = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                                     LSTM_units=128, backbone=False)
            elif args.backbone == 'SDCL':
                model = SDCL(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                             LSTM_units=128, backbone=False, **snn_params)
            elif args.backbone == 'SFCN':
                # CASNN shares weights with SFCN
                model = SFCN(n_channels=args.n_feature,
                             n_classes=args.n_class, backbone=False, **snn_params)
            elif args.backbone == 'SRESNET':
                # CASRESNET shares weights with SRESNET
                model = SRESNET(n_channels=args.n_feature,
                                n_classes=args.n_class, backbone=False, **snn_params)
            else:
                raise NotImplementedError

            model = model.to(DEVICE)
            save_dir = 'results/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            criterion = nn.CrossEntropyLoss()

            parameters = model.parameters()
            optimizer = torch.optim.Adam(parameters, args.lr)

            train_loss_list = []
            test_loss_list = []

            best_model = train(args, train_loaders, val_loader,
                               model, DEVICE, optimizer, criterion)
        
        # Testing
        else:
            if not os.path.exists('testloader/' + 'samples_' + args.dataset + '_' + args.target_domain + '.txt'):
                # this can save reapeted data process time and genearte test file for onnxruntime application.
                print(Fore.CYAN + 'No testloader files, preparing ...')
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
                if not os.path.exists('testloader/'):
                    os.makedirs('testloader/')                
                np.savetxt(f'./testloader/samples_{args.dataset}_{args.target_domain if (args.cases == "subject_large" or args.cases == "subject") else "_"}.txt', samples, fmt='%.4f', delimiter='\n')
                np.savetxt(f'./testloader/targets_{args.dataset}_{args.target_domain if (args.cases == "subject_large" or args.cases == "subject") else "_"}.txt', targets, fmt='%d', delimiter='\n')

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
            
            criterion = nn.CrossEntropyLoss()
            save_dir = 'results/'
            best_model = torch.load(save_dir + args.model_name + '.pt', map_location=DEVICE)['model_state_dict']
            if args.eescan:
                # train_loaders, val_loader, test_loader = setup_dataloaders(args)
                tee_scan(args, test_loader, best_model,
                         DEVICE, criterion, args.model_name)

        if args.backbone == 'FCN':
            model_test = FCN(n_channels=args.n_feature,
                             n_classes=args.n_class, backbone=False)
        elif args.backbone == 'RESNET':
            model_test = RESNET(n_channels=args.n_feature,
                              n_classes=args.n_class, backbone=False)
        elif args.backbone == 'DCL':
            model_test = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                                    LSTM_units=128, backbone=False)
        elif args.backbone == 'SDCL':
            model_test = SDCL(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5,
                            LSTM_units=128, backbone=False, **snn_params)
        elif args.backbone == 'SFCN':
            # model_test = SFCN_eval(n_channels=args.n_feature,
            #                   n_classes=args.n_class, backbone=False, **snn_params)
            
            # For acquiring spikes of every inference
            save_path = f'spikelog/{args.dataset}_{args.target_domain}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_test = SFCN_getspike(save_path=save_path, n_channels=args.n_feature,
                              n_classes=args.n_class, backbone=False, **snn_params)
        elif args.backbone == 'SRESNET':
            model_test = SRESNET(n_channels=args.n_feature,
                              n_classes=args.n_class, backbone=False, **snn_params)
        elif args.backbone == 'CASNN':
            model_test = CASNN(n_channels=args.n_feature, n_classes=args.n_class,
                               backbone=False, ee_thresh=args.tee,  **snn_params)
        else:
            raise NotImplementedError

        model_test.load_state_dict(best_model, strict=False)
        model_test = model_test.to(DEVICE)

        avgmeter = hook_layers(model_test)                

        test_loss = None
        tlasts = None
        if args.backbone == 'CASNN':
            test_loss, tlasts = test_ee(
                args, test_loader, model_test, DEVICE, criterion)
        else:
            test_loss, tlasts = test(
                args, test_loader, model_test, DEVICE, criterion, plt=False)
        acc_list.append(test_loss)
        print(Fore.YELLOW + f"Time last: {tlasts * 1000:.2f} ms")

        print("Fire Rate: {}".format(avgmeter.avg()))

    training_end = datetime.now()
    training_time = training_end - training_start
    print(f"Training time is : {training_time}")

    a = np.array(acc_list)
    print('Final Accuracy: {}, Std: {}'.format(np.mean(a), np.std(a)))

    # * Calculate FLOPs
    # flops_info(model_test, test_loader)
