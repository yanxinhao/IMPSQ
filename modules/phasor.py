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
from .fft_util import diff_rdft2


class Phasor(nn.Module):
    def __init__(
        self,
        resolutions,
        dim_feat,
        gauss_variance=0,
        num_comp_log=-1,
        TV_weight=0,
        device="cuda",
    ) -> None:
        """ 
            Paramters:
                res:  freqsize (freq resolution)
                dims: feature dim_feat
                num_comp: number of components
        """
        # resolutions = resolutions[:2]
        # resolutions = [256, 256]
        super(Phasor, self).__init__()
        self.res = torch.tensor(resolutions)
        self.TV_weight = TV_weight
        self.device = device
        # gaussion
        # self.gauss_variance = torch.nn.Parameter(
        #     torch.tensor([gauss_variance] * 2).to(device)
        # )
        self.gauss_variance = torch.tensor([gauss_variance] * 2).to(device)
        #
        if num_comp_log == -1:
            # num_comp_log = [math.ceil(np.log2(N)) + 1 for N in resolutions]
            num_comp_log = [math.ceil(np.log(N)) + 1 for N in resolutions]
        else:
            num_comp_log = [num_comp_log, num_comp_log]
        self.axis = [
            torch.tensor([0.0] + [2 ** i for i in torch.arange(d - 1)]).to(device)
            for d in num_comp_log
        ]
        # self.axis = [
        #     torch.tensor(
        #         [i for i in torch.arange(15)]
        #         + [2 ** i for i in torch.arange(4, d - 1)],
        #         dtype=torch.float,
        #     ).to(device)
        #     for d in num_comp_log
        # ]
        self.num_comp = [x.shape[0] for x in self.axis]
        # ----------------------
        # self.num_comp = [257, 257]  # [N // 4 for N in resolutions]
        # self.num_comp = [40, 40]  # [N // 4 for N in resolutions]
        # self.axis = [
        #     torch.arange(0, d, dtype=torch.float32, device=self.device)
        #     for d in self.num_comp
        # ]
        # ----------------------
        self.params = nn.ParameterList(
            self.init_(self.num_comp, self.res.long(), ksize=dim_feat, init_scale=1)
        )
        self.ktraj = self.compute_ktraj(self.axis, self.res)
        self.alpha = nn.Parameter(torch.ones(1, device=self.device))

    def compute_gaussian(self, variance):
        ktraj = self.ktraj
        gauss = [
            torch.exp((-2 * (np.pi * kk) ** 2 * variance[None]).sum(-1)).reshape(
                1, 1, *kk.shape[:-1]
            )
            for kk in ktraj
        ]
        return gauss

    def compute_ktraj(self, axis, res):
        ktraj2d = [torch.fft.fftfreq(i, 1 / i).to(self.device) for i in res]
        ktraj1d = [
            torch.arange(ax).to(torch.float).to(self.device) if type(ax) == int else ax
            for ax in axis
        ]
        ktrajx = torch.stack(torch.meshgrid([ktraj1d[0], ktraj2d[1]]), dim=-1)
        ktrajy = torch.stack(torch.meshgrid([ktraj2d[0], ktraj1d[1]]), dim=-1)
        ktraj = [ktrajx, ktrajy]
        return ktraj

    @property
    def kspace(self):
        gauss = self.compute_gaussian(self.gauss_variance)
        return (
            self.params[0] * gauss[0],  # * self.alpha,
            self.params[1] * gauss[1],  # * self.alpha,
        )

    def forward(self, inputs, variance=0, bound=1):
        # inputs = inputs[..., :2]
        assert inputs.shape[1] == 2
        inputs = inputs / bound  # map to [-1, 1]
        # assert inputs.max() <= 1 and inputs.min() >= -1

        feature = self.compute_fft(self.kspace, inputs, interp=False)
        # features = torch.concat(feature, dim=0)
        return feature.T

    def compute_fft(self, features, xyz_sampled, interp=True):
        # this is fast because we did 1d transform and matrix multiplication . (N*N) logN d + Nsamples * d*d + 3 * Nsamples
        Nx, Ny = self.res
        Fx, Fy = features
        d1, d2 = Fx.shape[2], Fy.shape[3]
        kx, ky = self.axis
        kx, ky = kx[:d1], ky[:d2]
        xs, ys = xyz_sampled.chunk(2, dim=-1)
        Fx = torch.fft.ifftn(Fx, dim=(3), norm="forward")
        Fy = torch.fft.ifftn(Fy, dim=(2), norm="forward")
        fake_zs = torch.zeros_like(xs)
        # interpolation
        fx = grid_sample(
            Fx.transpose(3, 3).flatten(1, 2).unsqueeze(-1),
            torch.stack([fake_zs, ys], dim=-1)[None],
        ).reshape(1, Fx.shape[1], Fx.shape[2], -1)
        fy = grid_sample(
            Fy.transpose(2, 3).flatten(1, 2).unsqueeze(-2),
            torch.stack([xs, fake_zs], dim=-1)[None],
        ).reshape(1, Fy.shape[1], Fy.shape[3], -1)
        fx, fy = fx.squeeze(0), fy.squeeze(0)
        # matrix
        fxx = batch_irfft(fx, xs, kx, Nx)
        fyy = batch_irfft(fy, ys, ky, Ny)

        return (fxx + fyy)#/2

    def init_(self, axis, res, ksize=1, init_scale=1):
        # transform the fourier domain to spatial domain
        # Fx, Fy, Fz = features
        Nx, Ny = res
        d1, d2 = axis
        xx, yy = [torch.linspace(0, 1, N) for N in (d1, d2)]
        kx, ky = self.axis
        kx, ky = kx[:d1], ky[:d2]
        fx = (
            torch.zeros(1, ksize, d1, Ny, device=self.device).to(torch.complex64)
            * init_scale
        )
        fy = (
            torch.zeros(1, ksize, Nx, d2, device=self.device).to(torch.complex64)
            * init_scale
        )
        # transform back to the fourier domain
        # fx = rfft(
        #     torch.fft.fftn(fx, dim=3, norm="forward").transpose(2, 3), xx, ff=kx, T=Nx
        # ).transpose(2, 3)
        # fy = rfft(
        #     torch.fft.fftn(fy, dim=2, norm="forward").transpose(3, 3), yy, ff=ky, T=Ny
        # ).transpose(3, 3)
        return [
            torch.nn.Parameter(fx),
            torch.nn.Parameter(fy),
        ]

    def TV_loss(self, order=1):
        ktrajx, ktrajy = self.ktraj
        new_phasor_fx = diff_rdft2(self.params[0], ff=ktrajx, order=order)
        new_phasor_fy = diff_rdft2(self.params[1], ff=ktrajy, order=order)
        tv_fx = self.TV_weight * new_phasor_fx.abs().square().mean()
        tv_fy = self.TV_weight * new_phasor_fy.abs().square().mean()
        loss = tv_fx + tv_fy
        return loss


def batch_irfft(phasors, xx, ff, T):
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx + 1) * 0.5
    xx = xx * (T - 1) / T
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j * np.pi * xx * ff)  # twiddle factor
    twiddle = twiddle * ((ff >= 0) + 1)[None]
    # twiddle[:,1:-1] = twiddle[:, 1:-1] * 2                    # hermitian # [N, d] # inplace operation
    twiddle = twiddle.transpose(0, 1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)


def rfft(spatial, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T - 1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1 / T)  # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out

