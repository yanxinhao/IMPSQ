import torch
import math
import numpy as np
import cv2
import matplotlib.colors as colors
import cmapy


def to_uint8(x):
    return (255.0 * x).astype(np.uint8)


def to_opencv_img(x):
    img = to_uint8(x)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def rescale_img(x, mode="scale", perc=None, tmax=1.0, tmin=0.0):
    if mode == "scale":
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif mode == "clamp":
        x = torch.clamp(x, 0, 1)
    return x


def grads2img(gradients):
    mG = gradients.detach().cpu()
    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2.0 * math.pi)
    mGhsv[:, :, 1] = 1.0

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    # return mGrgb
    rgb = to_opencv_img(mGrgb)
    return rgb


def laplace2img(laplace):
    laplace_img = to_uint8((rescale_img(laplace, "scale", 1)).detach().cpu().numpy())
    laplace_img = cv2.applyColorMap(laplace_img, cmapy.cmap("RdBu"))
    laplace_img = cv2.cvtColor(laplace_img, cv2.COLOR_BGR2RGB)
    return laplace_img


def grid_sample(image, grid, **kwargs):
    N, C, IH, IW = image.shape
    _, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    # pdb.set_trace()q
    image = image.reshape(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().reshape(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.reshape(N, C, H, W) * nw.reshape(N, 1, H, W)
        + ne_val.reshape(N, C, H, W) * ne.reshape(N, 1, H, W)
        + sw_val.reshape(N, C, H, W) * sw.reshape(N, 1, H, W)
        + se_val.reshape(N, C, H, W) * se.reshape(N, 1, H, W)
    )

    return out_val

