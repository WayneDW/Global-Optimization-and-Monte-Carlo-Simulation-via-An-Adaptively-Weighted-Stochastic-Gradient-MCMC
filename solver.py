#!/usr/bin/env python3
import sys

import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform

from time import time

# Credit to Zhunzhong07
#from utils import Bar

class Sampler:
    def __init__(self, Fclass, pars):

        self.fmin = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0}
        self.domain = {'f1': [-2.56, 5.12], 'f2': [-300, 600], 'f3': [-5, 10], 'f4': [-5, 10], 'f5': [-5, 10], 'f6': [-4, 5],
                       'f7': [-10, 10], 'f8': [-10, 10], 'f9': [-2.56, 5.12], 'f10': [-1, 1]}
        self.xmin = {'f1': [0]*20, 'f2': [0]*20, 'f3': [0]*20, 'f4': [1]*20, 'f5': [0]*20, 'f6': [0]*24, 
                     'f7': list(map(lambda i: 2.**(-(2.**i-2)/2.**i), range(1, 26))), 'f8': [1]*30, 'f9': [0]*30, 'f10': [0]*30}

        self.fname = 'f' + str(pars.fnum)
        self.Fclass = Fclass
        self.dim = len(self.xmin[self.fname])
        self.lr = pars.lr
        self.invT = pars.invT
        self.boundary = self.domain[self.fname]
        self.xinit = np.array(list(map(lambda x: uniform(self.boundary[0], self.boundary[1]), range(self.dim))))
        self.max_iters = int(pars.max_iters)
        self.stochastic_noise_scale = pars.noise_scale
        self.error = pars.error
        self.zeta = pars.zeta
        self.div = pars.div
        self.check = pars.check
        self.decay_min = pars.decay_min
        self.decay_lr = pars.decay_lr
        self.total_parts = pars.part
        # start to calc partion from (f_init - f_min) * loss_thres + f_min
        self.loss_thres = pars.loss_thres

    def in_domain(self, beta): return sum(map(lambda i: beta[i] < self.boundary[0] or beta[i] > self.boundary[1], range(self.dim))) == 0

    def stochastic_grad(self, beta):
        call_func_by_str = 'self.Fclass.' + self.fname
        grad = eval('grad(' + call_func_by_str + ')')
        return grad(beta) + normal(size=self.dim) * (self.boundary[1] - self.boundary[0]) / 1e3

    def eval_f(self, beta): return eval('self.Fclass.' + self.fname + '(' + str(beta.tolist()) + ')')


    def set_adaptive(self):
        self.J = self.total_parts - 1
        init_f = self.eval_f(self.xinit)
        self.Gcum = np.array([0.1] * self.total_parts)
        self.min_f = self.fmin[self.fname]

    def find_aw_idx(self, beta): return(min(max(int((self.eval_f(beta) - self.min_f) / self.div + 1), 1), self.total_parts - 1))

    def record(self, method, time_used, iters, finit, freal, fcalc):
        print('%s,%s, time %.1f'%(self.fname, method, time_used))
        print('%s,%s, temperature %.2f'%(self.fname, method, 1./self.invT))
        print('%s,%s, partition number %d'%(self.fname, method, self.total_parts))
        print('%s,%s, iter %d'%(self.fname, method, iters))
        print('%s,%s, init_f %.3f'%(self.fname, method, finit))
        print('%s,%s, true_fmin %.3f'%(self.fname, method, freal))
        print('%s,%s, estimated_f %.2f'%(self.fname, method, fcalc))
        print('=' * 60)



    def sgd(self):
        beta = self.xinit
        time_cnt = time()
        init_f = self.eval_f(beta)
        print('x0 in domain %s, f(x0) %.4f, global minima f %.4f'%(self.in_domain(beta), init_f, self.fmin[self.fname]))
        f_cur_min = self.eval_f(beta)
        for iters in range(1, int(self.max_iters+1)):
            proposal = beta - self.lr * self.stochastic_grad(beta)
            if self.in_domain(proposal):
                beta = proposal
            if self.eval_f(beta) < self.fmin[self.fname] + self.error:
                break
            if self.eval_f(beta) < f_cur_min:
                print('updated f %.4f'%(self.eval_f(beta)))
                f_cur_min = self.eval_f(beta)
        time_used = time() - time_cnt
        self.record('sgd', time_used, iters, init_f, self.fmin[self.fname], f_cur_min)


    def sgld(self):
        beta = self.xinit
        time_cnt = time()
        init_f = self.eval_f(beta)
        f_cur_min = init_f
        grad_scale, noise_scale = 0, 0
        #bar = Bar('Processing', max=self.max_iters/100)
        for iters in range(1, int(self.max_iters+1)):
            proposal = beta - self.lr * self.stochastic_grad(beta) + sqrt(2 * self.lr / self.invT) * normal(size=self.dim)
            if self.in_domain(proposal) and self.check:
                beta = proposal
            if self.check == 0:
                beta = proposal
            if self.eval_f(beta) < f_cur_min:
                f_cur_min = self.eval_f(beta)

            if iters % 100 == 0:
                suffix  = 'Domain ({left:.2f}, {right:.2f}) Dim:{dim: .0f} Initial f: {init_f:.3f} estimated f: {estimated_f:.3f} current f min: {cur_f:.3f} f optimal: {f_min:.3f} error: {error:.3f}'.format(\
                    left=self.boundary[0],\
                    right=self.boundary[1],\
                    dim=self.dim,\
                    init_f=init_f, \
                    estimated_f=self.eval_f(beta), \
                    cur_f=f_cur_min,\
                    f_min=self.fmin[self.fname], \
                    error=self.error)
                print(suffix)
            
            if self.eval_f(beta) < self.fmin[self.fname] + self.error:
                print('Optimization Completed')
                break
        time_used = time() - time_cnt
        self.record('SGLD', time_used, iters, init_f, self.fmin[self.fname], f_cur_min)

    def awsgld(self):
        beta = self.xinit
        time_cnt = time()
        init_f = self.eval_f(beta)
        f_cur_min = init_f
        grad_scale, noise_scale = 0, 0
        self.set_adaptive()
        for iters in range(1, int(self.max_iters+1)):
            grad_mul = 1 + self.zeta / self.invT * (np.log(self.Gcum[self.J]) - np.log(self.Gcum[self.J-1])) / self.div
            proposal = beta - self.lr * grad_mul * self.stochastic_grad(beta) + sqrt(2. * self.lr / self.invT) * normal(size=self.dim)
            decay = min(self.decay_min, self.decay_lr / (iters**0.75 + 1000.))
            if self.in_domain(proposal) and self.check:
                beta = proposal
            if self.check == 0:
                beta = proposal
            self.J = self.find_aw_idx(beta)
            self.Gcum[self.J: ] = self.Gcum[self.J: ] + decay * (self.Gcum[self.J]) * (1. - self.Gcum[self.J: ])
            self.Gcum[:self.J] = self.Gcum[:self.J] + decay * (self.Gcum[self.J] * (-self.Gcum[:self.J]))
            if iters % 3e3 == 0 or iters == 1000:
                print('\nDelta h ' +  str(self.div)[:4])
                grad_muls = 1 + (np.log(self.Gcum[1:]) - np.log(self.Gcum[:-1])) * self.zeta / self.div / self.invT
                print('\nGrad multiplier max ' + str(np.max(grad_muls))[:4])
                print(grad_muls)
                print('\n CDF in energy')
                print(self.Gcum)
            if self.eval_f(beta) < f_cur_min:
                f_cur_min = self.eval_f(beta)

            if iters % 100 == 0:
                suffix  = 'Domain ({left:.2f}, {right:.2f}) Dim:{dim: .0f} Initial f: {init_f:.3f} estimated f: {estimated_f:.3f} current f min: {cur_f:.3f} f optimal: {f_min:.3f} error: {error:.3f}'.format(\
                    left=self.boundary[0],\
                    right=self.boundary[1],\
                    dim=self.dim,\
                    init_f=init_f, \
                    estimated_f=self.eval_f(beta), \
                    cur_f=f_cur_min,\
                    f_min=self.fmin[self.fname], \
                    error=self.error)
                print(suffix)

            if self.eval_f(beta) < self.fmin[self.fname] + self.error:
                if iters <= 100:
                    suffix  = 'Domain ({left:.2f}, {right:.2f}) Initial f: {init_f:.3f} estimated f: {estimated_f:.3f} current f min: {cur_f:.3f} f optimal: {f_min:.3f} error: {error:.3f}'.format(\
                    left=self.boundary[0],\
                    right=self.boundary[1],\
                    init_f=init_f, \
                    estimated_f=self.eval_f(beta), \
                    cur_f=f_cur_min,\
                    f_min=self.fmin[self.fname], \
                    error=self.error)
                    print(suffix)
                break
        print('\nGrad multiplier')
        print(1 + (np.log(self.Gcum[1:]) - np.log(self.Gcum[:-1])) * self.zeta / self.div / self.invT)
        print('Max multiplier')
        print(max(1 + (np.log(self.Gcum[1:]) - np.log(self.Gcum[:-1])) * self.zeta / self.div / self.invT))
        time_used = time() - time_cnt
        self.record('AWSGLD', time_used, iters, init_f, self.fmin[self.fname], f_cur_min)
