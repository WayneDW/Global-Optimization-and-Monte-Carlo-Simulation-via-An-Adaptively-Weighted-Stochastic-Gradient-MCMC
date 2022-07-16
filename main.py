#!/usr/bin/env python3
import sys
import time
from solver import Sampler
from test_functions import Test_Functions

import numpy as np
import argparse
import datetime

Fclass = Test_Functions()

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-fnum', default=1, type=int, help='function number')
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-invT', default='1', type=float, help='inverse temperature')
parser.add_argument('-T', default='1', type=float, help='Temperature')
parser.add_argument('-method', default='sgld', type=str, help='SGLD or AWSGLD')
parser.add_argument('-zeta', default=1, type=float, help='Adaptive hyperparameter')
parser.add_argument('-error', default=1e-3, type=float, help='Acceptable error/ error budget')
parser.add_argument('-div', default=1, type=float, help='Energy in each region')
parser.add_argument('-check', default=0, type=int, help='Domain check')
parser.add_argument('-max_iters', default=1e5, type=float, help='Training iters')
parser.add_argument('-noise_scale', default=0.1, type=float, help='stochastic gradient scale')
parser.add_argument('-decay_min', default=1, type=float, help='Decay starting rate')
parser.add_argument('-decay_lr', default=100, type=float, help='Decay lr')
parser.add_argument('-part', default=100, type=int, help='Total part')
pars = parser.parse_args()

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

pars.fnum = pars.fnum

pars.invT = 1. / pars.T

sampler = Sampler(Fclass, pars)

sampler_low = Sampler(Fclass, pars)
sampler_low.invT *= 10

if pars.method == 'sgld':
    print(f'Start running high-temperature SGLD')
    sampler.sgld()
    print(f'Start running low-temperature SGLD')
    sampler_low.sgld()
elif pars.method == 'awsgld':
    print(f'Start running AWSGLD')
    sampler.awsgld()

