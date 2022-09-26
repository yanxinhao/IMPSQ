import copy
import numpy as np
import torch
import torch.nn.functional as F
from .image_util import grid_sample
import torch.autograd as autograd
from torch.autograd import Function
from .diff_operators import gradient


def time2freq(gridSize):
    ffSize = copy.deepcopy(gridSize)
    ffSize[-1] = ffSize[-1] // 2 + 1
    return ffSize


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


def idft(phasors, x, Nx, fk=None, dim=-1):
    # x should be in [0,1]                      # F(f(ax)) = 1/|a| P(w/a)
    # x = x * (Nx - 1) / Nx
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    N = phasors.shape[-1]  # frequency domain scaling
    if fk is None:
        pf = torch.arange(0, (N + 1) // 2).to(device)  # positive freq
        nf = torch.arange(-(N - 1) // 2, 0).to(device)  # negative freq
        fk = torch.cat([pf, nf])  # sampling frequencies
    x = x.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * x * fk).to(device)
    out = F.linear(phasors, M)  # integrate phasors
    out = out.transpose(dim, -1)  # transpose back
    return out


def irdft(phasors, x, Nx, fk=None, dim=-1):
    # x = x * (Nx - 1) / Nx
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    N = phasors.shape[-1]
    if fk is None:
        pf = torch.arange(N).to(device)  # positive freq only
        fk = pf  # sampling frequencies
    x = x.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * x * fk).to(device)
    # index in pytorch is slow
    # M[:, 1:] = M[:, 1:] * 2                          # Hermittion symmetry
    M = M * ((fk > 0) + 1)[None]
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


class IRDFT2D(Function):
    @staticmethod
    def forward(
        ctx, phasors, coord, gridSize, freq_x, freq_y, dim=torch.tensor([0, 1])
    ):
        ctx.save_for_backward(phasors, coord, gridSize, freq_x, freq_y, dim)
        out = irdft2d(phasors, coord, gridSize, freq=[freq_x, freq_y], dim=dim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        phasors, coord, gridSize, freq_x, freq_y, dim = ctx.saved_tensors
        device = phasors.device
        # gradient wrt coord
        freq = torch.stack(torch.meshgrid([freq_x, freq_y]), dim=-1).to(device)
        new_phasor = diff_rdft2(phasors, gridSize, ff=freq, order=1)
        # TODO:
        grad_x = (
            irdft2d(new_phasor[..., 0], coord, gridSize, freq=[freq_x, freq_y], dim=dim)
            * grad_output
        ).sum(-1)
        grad_y = (
            irdft2d(new_phasor[..., 1], coord, gridSize, freq=[freq_x, freq_y], dim=dim)
            * grad_output
        ).sum(-1)
        grad_coord = torch.stack([grad_x, grad_y], dim=-1)
        return None, grad_coord, None, None, None, None


def irdft2d(phasors, coord, gridSize, freq=None, dim=[0, 1]):
    """irdft2 for 

    Args:
        phasors (_type_): _description_
        coord (_type_): _description_
        gridSize (_type_): _description_
        dim (list, optional): _description_. Defaults to [0, 1].

    Returns:
        _type_: _description_
    """
    assert coord.shape[-1] == 2
    assert coord.max() <= 1 and coord.min() >= -1
    device = phasors.device
    ifft_crop = phasors
    Nx, Ny = gridSize
    freq_x, freq_y = freq
    if freq_y is not None:
        assert freq_y.min() >= 0
    # grid points for ifft
    xx, yy = [torch.arange(N).to(device) for N in gridSize]
    ifft_crop = idft(ifft_crop, xx, Nx, freq_x, dim=dim[0])
    ifft_crop = irdft(ifft_crop, yy, Ny, freq_y, dim=dim[1])
    # return ifft_crop.reshape(-1, 1)
    # TODO: align corner
    out = grid_sample(ifft_crop, coord.flip(-1)[None, :, None, :], align_corners=True,)
    out = out.squeeze(0).squeeze(-1)
    return out.T


def diff_rdft2(phasors, gridSize=None, ff=None, order=1):
    device = phasors.device
    if ff is None:
        Nx, Ny = gridSize
        fx = torch.fft.fftfreq(Nx)
        fy = torch.fft.rfftfreq(Ny)
        freq = torch.stack(torch.meshgrid([fx, fy]), dim=-1).to(device)
    elif ff.shape[0] == 2:
        fx, fy = ff
        freq = torch.stack(torch.meshgrid([fx, fy]), dim=-1).to(device)
    else:
        freq = ff
    ch, dim, h, w = phasors.shape
    grad_phasor = phasors.reshape([ch * dim, h, w, 1]) * (
        freq[None, :, :, :] * 2 * np.pi * 1j
    ).pow(order)
    grad_phasor = grad_phasor.reshape([ch, dim, h, w, -1])
    return grad_phasor
