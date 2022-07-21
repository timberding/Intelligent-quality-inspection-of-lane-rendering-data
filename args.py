# -*- coding: UTF-8 -*-
import argparse
import os
import torch


def build_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--traindir', type=str, default=None)
    parser.add_argument('--train_metadir', type=str, default=None)
    parser.add_argument('--train_metafile', type=str, default=None)

    parser.add_argument('--testdir', type=str, default=None)
    parser.add_argument('--test_metadir', type=str, default=None)
    parser.add_argument('--test_metafile', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=0.9)

    opt = parser.parse_args()

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.resultdir = './result'
    if not os.path.exists(opt.resultdir):
        os.makedirs(opt.resultdir)

    print('.' * 75)
    for key in opt.__dict__:
        param = opt.__dict__[key]
        print('...param: {}: {}'.format(key, param))
    print('.' * 75)

    return opt


