##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, optimizer, mode, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0, min_lr=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.optimizer = optimizer
        self.lr = self.optimizer.defaults['lr']
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.min_lr = min_lr

    def __call__(self, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if T <= self.N:
            if self.mode == 'cos':
                lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            elif self.mode == 'poly':
                lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
            elif self.mode == 'step':
                lr = self.lr * (0.1 ** (epoch // self.lr_step))
            else:
                raise NotImplemented
        else:
            lr = 0
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        lr = max(lr, self.min_lr)
        self._adjust_learning_rate(lr)

    def _adjust_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def simulate_x_epochs(self, epoch):
        self.__call__(0, epoch)

