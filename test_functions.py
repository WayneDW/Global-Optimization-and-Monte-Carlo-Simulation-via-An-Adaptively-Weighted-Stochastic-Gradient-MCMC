#!/usr/bin/env python3

import autograd.numpy as np
from autograd import grad

from autograd.numpy import sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal


class Test_Functions:
    def __init__(self):

        self.error = 1e-3
        self.fmin = {'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0, 'f9': 0, 'f10': 0}
        self.xmin = {'f1': [0]*20, 'f2': [0]*20, 'f3': [0]*20, 'f4': [1]*20, 'f5': [0]*20, 'f6': [0]*24, 
                     'f7': list(map(lambda i: 2.**(-(2.**i-2)/2.**i), range(1, 26))), 'f8': [1]*30, 'f9': [0]*30, 'f10': [0]*30}
        self.domain = {'f1': [-2.56, 5.12], 'f2': [-300, 600], 'f3': [-5, 10], 'f4': [-5, 10], 'f5': [-5, 10], 'f6': [-4, 5], 
                       'f7': [-10, 10], 'f8': [-10, 10], 'f9': [-2.56, 5.12], 'f10': [-15, 30]}

    def f1(self, x): return 10*20 + sum(map(lambda i: x[i]**2 - 10*cos(2*pi*x[i]), range(20)))

    def f2(self, x): return sum(list(map(lambda i: x[i]**2/4000., range(20)))) - prod(list(map(lambda i: cos(x[i]/sqrt(i+1)), range(20)))) + 1

    def f3(self, x): return sum(map(lambda i: (i+1)*x[i]**2, range(20)))

    def f4(self, x): return sum(map(lambda i: 100 * (x[i]**2 - x[i+1])**2 + (x[i] - 1)**2, range(19)))

    def f5(self, x): return sum(map(lambda i: x[i]**2, range(20))) + sum(map(lambda i: i*x[i]/2., range(20)))**2 + sum(map(lambda i: i*x[i]/2., range(20)))**4

    def f6(self, x): return sum(map(lambda i: (x[4*i-3]+10*x[4*i-2])**2+5*(x[4*i-1]-x[4*i])**2+(x[4*i-2]-x[4*i-1])**4+10*(x[4*i-3]-x[4*i])**4, range(6)))

    def f7(self, x): return (x[0] - 1)**2 + sum(map(lambda i: i * (2*x[i]**2 - x[i-1])**2, range(1, 25)))

    def f8(self, x): return sin(pi * x[0])**2 + sum(map(lambda i: ((x[i]-1)**2*(1+10*sin(pi*x[i]+1)**2)), range(29))) + (x[29]-1)**2*(1+10*sin(2*pi*x[29])**2)

    def f9(self, x): return sum(map(lambda i: x[i]**2, range(30)))

    def f10(self, x): return 20 + exp(1) - 20 * exp(-0.2 * sqrt(sum(map(lambda i: x[i]**2, range(30)))/30.)) - exp(1./30 * sum(map(lambda i: cos(x[i]*2*pi), range(30))))

