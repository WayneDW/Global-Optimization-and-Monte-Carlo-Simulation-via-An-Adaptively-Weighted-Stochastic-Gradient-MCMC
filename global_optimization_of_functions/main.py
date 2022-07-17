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
parser.add_argument('-method', default='sgld', type=str, help='SGLD or AWSGLD')
""" general hyperparameters """
parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-T', default='1', type=float, help='Temperature')
""" AWSGLD: energy partition """
parser.add_argument('-div', default=1, type=float, help='Energy in each region')
parser.add_argument('-part', default=100, type=int, help='Total part')
""" AWSGLD: hyperparameters """
parser.add_argument('-zeta', default=1, type=float, help='Adaptive hyperparameter')
parser.add_argument('-decay_lr', default=100, type=float, help='Decay lr (step size for stochastic approximation)')

""" other """
parser.add_argument('-error', default=1e-3, type=float, help='Acceptable error/ error budget')
parser.add_argument('-check', default=0, type=int, help='Domain check (ensure the particle is in the set of interests)')
parser.add_argument('-max_iters', default=1e5, type=float, help='Training iters')
pars = parser.parse_args()

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

pars.fnum = pars.fnum

sampler = Sampler(Fclass, pars)

if pars.method == 'sgld':
    print(f'Start running high-temperature SGLD')
    sampler.sgld()
    print(f'Start running low-temperature SGLD')
    sampler_low = Sampler(Fclass, pars)
    print('Adopt a low temperature')
    sampler_low.T /= 10
    sampler_low.sgld()
elif pars.method == 'awsgld':
    print(f'Start running AWSGLD')
    sampler.awsgld()

