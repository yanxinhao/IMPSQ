# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2022-05-14 12:07:23
LastEditors: yanxinhao
FilePath: /PhasorImage/phasor_sample.py
Date: 2022-05-14 12:07:23
Description: 
"""
# rewrite phasor as an encoder for easy deployment
import math
from multiprocessing import dummy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
import numpy as np
import pdb
from .image_util import grid_sample
from .fft_util import time2freq, irdft2d, IRDFT2D, diff_rdft2


class PhasorSample(nn.Module):
    def __init__(
        self,
        dim_feat,
        freqSize=[100, 100],
        freq_log=[False, False],
        gridSize=[512, 512],
        gaussion_var=1e-3,
        TV_weight=0,
        device="cuda",
    ) -> None:
        super(PhasorSample, self).__init__()
        self.ch = 1
        self.freqSize = torch.tensor(freqSize)
        self.freq_log = freq_log
        self.dim_feat = dim_feat
        self.device = device
        self.gaussion_var = torch.tensor(gaussion_var, device=self.device)
        self.gridSize = torch.tensor(gridSize)
        self.TV_weight = TV_weight
        self.init_()
        self.compute_freq()

    def compute_gaussian(self, variance):
        ktraj = self.ktraj
        gauss = torch.exp((-2 * (np.pi * ktraj) ** 2 * variance[None]).sum(-1)).reshape(
            1, 1, *ktraj.shape[:-1],
        )
        return gauss

    @property
    def kspace(self):
        gauss = self.compute_gaussian(self.gaussion_var)
        return self.params * gauss

    def compute_freq(self):
        Nfx, Nfy = [x.item() for x in self.freqSize]
        Nx, Ny = [x.item() for x in self.gridSize]
        log_fx, log_fy = self.freq_log
        # for ifft
        if log_fx:
            pf = (
                torch.tensor([0.0] + [2 ** i for i in torch.arange(Nfx // 2)]).to(
                    self.device
                )
                / Nx
            )
            nf = (
                torch.tensor(
                    [-(2 ** i) for i in torch.arange((Nfx + 1) // 2 - 1).flip(0)]
                ).to(self.device)
                / Nx
            )
            self.fx = torch.cat([pf, nf])
        else:
            pf = torch.arange(0, (Nfx + 1) // 2).to(self.device) / Nx  # positive freq
            nf = torch.arange(-(Nfx - 1) // 2, 0).to(self.device) / Nx  # negative freq
            self.fx = torch.cat([pf, nf])  # sampling frequencies
        # for irfft
        if log_fy:
            self.pfy = (
                torch.tensor([0.0] + [2 ** i for i in torch.arange(Nfy - 1)]).to(
                    self.device
                )
                / Ny
            )
            nf = (
                torch.tensor([-(2 ** i) for i in torch.arange(Nfy - 1).flip(0)]).to(
                    self.device
                )
                / Ny
            )
            self.fy = torch.cat([self.pfy, nf])
        else:
            self.pfy = torch.arange(Nfy).to(self.device) / Ny
            nf = torch.arange(-(Nfy - 1), 0).to(self.device) / Ny  # negative freq
            self.fy = torch.cat([self.pfy, nf])
        self.freq = [self.fx, self.fy]
        # self.freq_h = torch.stack([self.fx, self.pfy], dim=0)
        self.freq_h = [self.fx, self.pfy]
        self.ktraj = torch.stack(torch.meshgrid([self.fx, self.pfy]), dim=-1)
        # self.freq_h = [self.fx, self.pfy]

    def forward(self, coord, kspace=None, training=True):
        if kspace is None:
            kspace = self.kspace
        if training:
            feat = irdft2d(
                kspace,
                coord.reshape(-1, 2),
                self.gridSize,
                self.freq_h,
                torch.tensor([2, 3]),
            )
        else:
            feat = IRDFT2D.apply(
                kspace,
                coord.reshape(-1, 2),
                self.gridSize,
                self.fx,
                self.pfy,
                torch.tensor([2, 3]),
            )
        return feat

    def init_(self):
        self.params = torch.nn.Parameter(
            torch.zeros(1, self.dim_feat, *self.freqSize)
            .to(torch.complex64)
            .to(self.device)
        )

    def TV_loss(self, order=1):
        new_phasor = diff_rdft2(self.params, self.gridSize, ff=self.ktraj, order=order)
        loss = self.TV_weight * new_phasor.abs().square().mean()
        return loss
