import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable

class Sampler:
    def __init__(self, net, pars):
        self.net = net
        self.lr = pars.lr
        self.T = pars.T
        self.wdecay = pars.wdecay
        self.N = pars.N
        """ Adaptive weighted """
        self.div = pars.div
        self.part = pars.part
        self.zeta = pars.zeta
        self.J = pars.part - 1
        self.gmul = 1.

        """ preconditioner """
        self.batch = pars.batch
        self.alpha = 0.01
        self.preg = 1e-3
        self.Fisher_info = []
        for param in net.parameters():
            p = torch.ones_like(param.data)
            self.Fisher_info.append(p)


        """ SG-HMC """
        self.V = 0.1
        self.beta = 0.5 * self.V * self.lr
        self.momentum = 0.9
        self.velocity = []        
        for param in net.parameters():
            p = torch.zeros_like(param.data)
            self.velocity.append(p)

        self.G = torch.tensor(np.array(range(1, pars.part+1)) * 1.0 / pars.part).cuda()
    
    def update_noise(self):
        return np.sqrt(2.0 * self.lr * self.T)

    def update_noise_HMC(self):
        self.sigma = np.sqrt(2.0 * self.lr * (1 - self.momentum - self.beta))
        return self.sigma * np.sqrt(self.T)


    def init_momentum(self):
        self.momentum = []
        for param in self.net.parameters():
            random_init = torch.cuda.FloatTensor(param.data.size()).normal_()
            self.momentum.append(random_init)

    def step(self, x, y, init=False):
        if init:
            self.velocity = []
            for param in self.net.parameters():
                random_init = torch.cuda.FloatTensor(param.data.size()).normal_() * 0.01
                #random_init = torch.zeros_like(param.data)
                self.velocity.append(random_init)
        
        noise_scale = self.update_noise_HMC()

        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            self.velocity[i].mul_(self.momentum).add_(grads, alpha=-self.lr).add_(proposal)
            param.data.add_(self.velocity[i])

    def cstep(self, x, y):
        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            param.data.add_(grads, alpha=-self.lr).add_(proposal)


    def pstep(self, x, y):
        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data

            self.Fisher_info[i].mul_(1 - self.alpha).add_(self.alpha * ((grads/self.batch)**2))
            preconditioner = torch.div(1., self.preg + torch.sqrt(self.Fisher_info[i]))
            grads.mul_(preconditioner)
            proposal.mul_(torch.sqrt(preconditioner))

            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            param.data.add_(grads, alpha=-self.lr).add_(proposal)

    def awstep(self, x, y, decay, loss):
        """ Update energy PDFs """
        self.J = int(np.clip((loss) / self.div, 1, self.part-1))
        self.gmul = 1 + self.zeta * self.T * (torch.log(self.G[self.J]) - torch.log(self.G[self.J-1])) / self.div
        self.randomField = -self.G[self.J] * self.G
        self.randomField[self.J:] = self.G[self.J] * (1. - self.G[self.J:])
        self.G = self.G + decay * self.randomField

        noise_scale = self.update_noise()
        for i, param in enumerate(self.net.parameters()):
            proposal = torch.cuda.FloatTensor(param.data.size()).normal_().mul(noise_scale)
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            grads.mul_(self.gmul)
            param.data.add_(grads, alpha=-self.lr).add_(proposal)
        return self.G[self.J].item() if self.J < self.part-1 else 0.

    def sgd(self, x, y):
        for i, param in enumerate(self.net.parameters()):
            grads = param.grad.data
            if self.wdecay != 0:
                grads.add_(param.data, alpha=self.wdecay)
            param.data.add_(grads, alpha=-self.lr)

