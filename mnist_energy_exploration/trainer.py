#!/usr/bin/python
import math
import copy
import sys
import os
import timeit
import csv
import dill
import argparse
import pickle
import random
from random import shuffle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

## import pytorch modules
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

## Import helper functions
from tools import BayesEval, StochasticWeightAvg
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()


def sgmcmc(net, train_loader, test_loader, pars):
    start = timeit.default_timer()
    samplers, BMA = [], BayesEval(pars.classes)
    criterion = nn.CrossEntropyLoss()
    sampler = Sampler(net, pars)

    import_w = 0

    counter = 0
    for epoch in range(pars.sn):

        if pars.c == 'cyc':
            sub_sn = pars.sn / 10
            cur_beta = (epoch % sub_sn) * 1.0 / sub_sn
            """ constant learning rate during exploration """
            sampler.lr = pars.lr / 2 * (np.cos(np.pi * min(cur_beta, 0.8)) + 1)
            if (epoch % sub_sn) * 1.0 / sub_sn == 0:
                print('Cooling down for optimization')
                sampler.T /= 1e10
            elif epoch % sub_sn == int(0.8 * sub_sn):
                print('Heating up for sampling')
                sampler.T *= 1e10

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)

            counter += 1

            filters = (labels < pars.classes)
            images, labels = images[filters], labels[filters]
            net.train()
            sampler.net.zero_grad()
            """ one class has roughly 6000 images """
            loss = criterion(sampler.net(images), labels) * pars.classes * 6000
            loss.backward()
            if pars.c == 'awsghmc':
                import_w = sampler.awstep(images, labels, pars.stepsize, loss.item())
            elif pars.c == 'cyc':
                sampler.cstep(images, labels)
            elif pars.c == 'sghmc':
                if counter % 500 == 0:
                    print('Reset momentum')
                    sampler.step(images, labels, init=True)
                else:
                    sampler.step(images, labels, init=False)
            elif pars.c == 'psgld':
                sampler.pstep(images, labels)
        """ apply weight 1 using SGHMC and importance weights using Contour SGHMC """
        if ((pars.c in ['sghmc', 'awsghmc', 'psgld'] and epoch >= 0.5 * pars.sn) or (pars.c == 'cyc' and cur_beta > 0.8)) and pars.classes != 2:
            BMA.eval(net, train_loader, test_loader, criterion, pars.classes, weight=import_w, bma=True)
        else:
            BMA.eval(net, train_loader, test_loader, criterion, pars.classes, weight=import_w, bma=False)

        Gcum = sampler.G.cpu().numpy()
        print('Epoch {} Acc: {:0.2f} BMA: {:0.2f} lr: {:.2E} T: {:.2E}  Weight {:.3f} Grad mul {:.2f} Pidx {} train Loss: {:0.1f} test Loss: {:0.1f}'.format(\
            epoch, BMA.cur_acc,  BMA.bma_acc, sampler.lr,  sampler.T, import_w, sampler.gmul, sampler.J, BMA.train_loss, BMA.test_loss))
       
    end = timeit.default_timer()
    print("Sampling Time used: {:0.1f}".format(end - start))
