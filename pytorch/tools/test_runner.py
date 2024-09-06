
# encoding=utf-8
from main import plot_dir_name
import matplotlib.pyplot as plt
from models.spike import *
from trainer import *
import torch
from datetime import datetime
import time
import numpy as np
from utils import _logger, hook_layers, tsne, mds
from colorama import init, Fore, Back, Style
init(autoreset=True)

def test(args, test_loader, model, DEVICE, criterion, plt=False):
    with torch.no_grad():
        tlasts = []
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0
        feats = None
        prds = None
        trgs = None
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        for idx, (sample, target, domain) in enumerate(test_loader):
            n_batches += 1
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()
            sample = sample.transpose(1, 2)

            # * Inference time
            t0 = time.time()
            out, features = model(sample)
            t1 = time.time()
            tlasts.append(t1 - t0)

            loss = criterion(out, target)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
            if prds is None:
                prds = predicted
                trgs = target
                feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                feats = torch.cat((feats, features), 0)
            trgs = torch.cat((trgs, target))
            feats = torch.cat((feats, features), 0)

        tlasts = np.array(tlasts).sum()
        acc_test = float(correct) * 100.0 / total

    print(n_batches)
    print(f'Test Loss     : {total_loss / n_batches:.4f}\t | \tTest Accuracy     : {acc_test:2.4f}\n')
    for t, p in zip(trgs.view(-1), prds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    if plt == True:
        tsne(feats, trgs, domain=None, save_dir=plot_dir_name +
             args.model_name + '_tsne.png')
        mds(feats, trgs, domain=None, save_dir=plot_dir_name +
            args.model_name + 'mds.png')
        sns_plot = sns.heatmap(confusion_matrix, cmap='Blues', annot=True)
        sns_plot.get_figure().savefig(plot_dir_name + args.model_name + '_confmatrix.png')
    return acc_test, tlasts


def test_ee(args, test_loader, model, DEVICE, criterion, prt_cmat=False, save_log=True, ee_logger=None):
    inflogs = []
    with torch.no_grad():
        model.eval()
        total_loss = 0
        n_batches = 0
        total = 0
        correct = 0

        feats = None
        prds = None
        trgs = None
        confusion_matrix = torch.zeros(args.n_class, args.n_class)
        tlasts = []

        # ! FOR EARLY EXIT
        prev_target = None
        prev_out = None
        prev_features = None

        for idx, (sample, target, domain) in enumerate(test_loader):
            n_batches += 1
            ee_pos = -1
            sample, target = sample.to(
                DEVICE).float(), target.to(DEVICE).long()

            sample = sample.transpose(1, 2)  # sample: [1,128,9] -> [1,9,128]

            # * Inference time
            t0 = time.time()
            out, features = model(sample)
            t1 = time.time()
            tlasts.append(t1 - t0)

            # ! FOR EARLY EXIT
            if out is None:
                ee_pos = features
                out = prev_out
                features = prev_features
            else:
                prev_out = out
                prev_features = features
                prev_target = target
            # ! FOR EARLY EXIT

            loss = criterion(out, target)
            total_loss += loss.item()
            _, predicted = torch.max(out.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum()

            # * Inference Logs
            inflog = [0, n_batches, predicted.numpy()[0], target.numpy()[
                0], bool(predicted == target), ee_pos]
            inflogs.append(inflog)

            # * confusion matrix
            if prds is None:
                prds = predicted
                trgs = target
                feats = features[:, :]
            else:
                prds = torch.cat((prds, predicted))
                trgs = torch.cat((trgs, target))
                feats = torch.cat((feats, features), 0)
        tlasts = np.array(tlasts).sum()
        acc_test = float(correct) * 100.0 / total

    print(
       Fore.MAGENTA +  f'Test Loss: {total_loss / n_batches:.4f}\t | \tTest Accuracy: {acc_test:2.4f}')
    if save_log:
        # Mark Early Exit Points
        np.save(f'spikelog/inflog/{args.dataset}_{args.target_domain}_{args.tee}.npy',
                np.array(inflogs))

    ee = np.array(inflogs)[:, -2:]
    ee_count = []
    for i in range(-1, 3):   # -1: None ee, 0-2: ee points from shallow to deep
        ee_pointi = ee[ee[:, -1] == i][:, 0]
        # Number of correct ee (True=1) and incorrect ee (False=0)
        ee_count.append([ee_pointi[ee_pointi == 1].shape[0],
                        ee_pointi[ee_pointi == 0].shape[0]])
    ee_count.append([acc_test, 0])
    print(Fore.MAGENTA + f'Early exit info: ' + str(ee_count[:-1]) + f'\t[correct,error](None, ee at L1-3)')
    if ee_logger is not None:
        ee_logger.append(ee_count)
    if prt_cmat:
        for t, p in zip(trgs.view(-1), prds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        print(confusion_matrix)
        print(confusion_matrix.diag() / confusion_matrix.sum(1))
    return acc_test, tlasts
