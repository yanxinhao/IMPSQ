# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2022-05-14 15:27:10
LastEditors: yanxinhao
FilePath: /PhasorImage/image.py
Date: 2022-05-14 15:27:01
Description: 
"""
import torch
import torch.nn as nn
import numpy as np
import os
from .phasor_sample import PhasorSample

import cv2
from .net import FCBlock, PosEncodingNeRF
from .diff_operators import gradient, laplace
from .image_util import grads2img, laplace2img
import scipy.ndimage
from .grid import Grid
import torch.nn.functional as F
from .image_util import grads2img, laplace2img
from .diff_operators import gradient
from .phasor import Phasor


class ImagePhasor(nn.Module):
    def __init__(
        self,
        H,
        W,
        ch=3,
        using_mlp=True,
        feat_mode="phasor",
        feat_cfg=None,
        mlp_cfg=None,
        device="cuda",
    ):
        super(ImagePhasor, self).__init__()
        self.feat_mode = feat_mode
        self.feat_cfg = feat_cfg
        self.mlp_cfg = mlp_cfg
        self.using_mlp = using_mlp
        self.device = device
        self.ch = ch
        self.H, self.W = H, W
        # init feature space
        self.initialize()

    def initialize(self):
        if self.using_mlp:
            self.mlp = FCBlock(**self.mlp_cfg).to(self.device)
            self.dim_feat = self.mlp_cfg.in_features

        else:
            self.dim_feat = self.ch
        if self.feat_mode == "phasor":
            self.feat = Phasor(**self.feat_cfg, device=self.device)
        elif self.feat_mode == "phasor_sample":
            self.feat = PhasorSample(
                **self.feat_cfg, gridSize=[self.H, self.W], device=self.device
            )
        elif self.feat_mode == "grid":
            self.feat = Grid(**self.feat_cfg)
        elif self.feat_mode == "pe":
            self.feat = PosEncodingNeRF(**self.feat_cfg)
        else:
            raise ValueError('feat mode must be in "phasor" or "grid"]')
        # init coord
        image_size = [self.H, self.W]
        self.xx, self.yy = [torch.arange(N).float().to(self.device) for N in image_size]
        self.coord = torch.stack(torch.meshgrid([self.xx, self.yy]), dim=-1)
        self.norm_xx, self.norm_yy = [
            torch.linspace(-1, 1, N).float().to(self.device) for N in image_size
        ]
        self.norm_coord = torch.stack(
            torch.meshgrid([self.norm_xx, self.norm_yy]), dim=-1
        )

    def _rectify_(self, coord):
        h, w = self.H, self.W
        recf_coord_x = 2 * coord[..., 0] / (h - 1) - 1
        recf_coord_y = 2 * coord[..., 1] / (w - 1) - 1
        return torch.stack([recf_coord_x, recf_coord_y], dim=-1)

    def compute_feat(self, coord, **kwargs):
        """sample feature

        Args:
            coord (tenspr): [n,2]

        Raises:
            ValueError: _description_

        Returns:
            feature: _description_
        """
        if self.feat_mode == "phasor":
            feat = self.feat(coord)
            return feat
        elif self.feat_mode == "phasor_sample":
            kspace = kwargs.get("kspace")
            training = kwargs.get("training", True)
            feat = self.feat(coord, kspace, training=training)
            return feat
        elif self.feat_mode == "grid":
            return self.feat(coord)
        elif self.feat_mode == "pe":
            return self.feat(coord).squeeze(1)
        else:
            raise ValueError('feat mode must be in "phasor" or "grid"]')

    @property
    def rendering(self):
        out = self.forward()
        img = np.array(255.0 * (out.detach().cpu().numpy() + 1) / 2, dtype=np.uint8)
        return img

    def forward(self, coord=None, rectify_coord=False, **kwargs):
        if coord is None:
            coord = self.norm_coord
        elif rectify_coord == True:
            coord = self._rectify_(coord)
        n = coord.shape[0]
        feat = self.compute_feat(coord.reshape(-1, 2), **kwargs)
        if self.using_mlp:
            out = self.mlp(feat)
        else:
            out = feat
        # out = F.tanh(out)
        out = out.reshape([n, -1])
        return out

    @property
    def grad(self, chunk_row=128):
        # if coord is None:
        torch.cuda.empty_cache()
        coord = nn.Parameter(self.coord)
        gradients = []
        for x in torch.split(coord, chunk_row, dim=0):
            out = self.forward(self._rectify_(x), training=False)
            pred_gradients = gradient(out, x).detach()
            gradients.append(pred_gradients)
            torch.cuda.empty_cache()
        gradients = torch.cat(gradients, dim=0)
        grid_img = grads2img(gradients)
        return grid_img, gradients

    @property
    def laplace(self, chunk_row=64):
        # if coord is None:
        torch.cuda.empty_cache()
        coord = nn.Parameter(self.coord)
        laplace = []
        for x in torch.split(coord, chunk_row, dim=0):
            out = self.forward(self._rectify_(x), training=False)
            pred_laplace = laplace(out, x).detach()
            laplace.append(pred_laplace)
            # del out, pred_laplace
            # gc.collect()
            torch.cuda.empty_cache()
        laplace = torch.cat(laplace, dim=0)
        laplace_img = laplace2img(laplace)
        return laplace_img, laplace

    def save_res(self, save_folder, res_dict={}):
        res_dict["model_state_dict"] = self.state_dict()
        torch.save(res_dict, os.path.join(save_folder, "model.pth"))
        gt_path = os.path.join(save_folder, "gt.png")
        cv2.imwrite(gt_path, self.image)
        # -------
        pred_img_path = os.path.join(save_folder, "res.png")
        cv2.imwrite(pred_img_path, self.rendering)
        # ----
        gt_grad_path = os.path.join(save_folder, "grad_gt.png")
        cv2.imwrite(gt_grad_path, grads2img(self.grad_gt))
        # ----
        pred_grad_path = os.path.join(save_folder, "grad.png")
        pred_grad = self.grad[0]
        cv2.imwrite(pred_grad_path, pred_grad)
        # ----
        gt_laplace_path = os.path.join(save_folder, "laplace_gt.png")
        cv2.imwrite(gt_laplace_path, laplace2img(self.laplace_gt))
        # ----
        pred_laplace_path = os.path.join(save_folder, "laplace.png")
        pred_laplace = self.laplace[0]
        cv2.imwrite(pred_laplace_path, pred_laplace)

    @property
    def grad(self, chunk_row=128):
        # if coord is None:
        torch.cuda.empty_cache()
        coord = nn.Parameter(self.coord)
        gradients = []
        for x in torch.split(coord, chunk_row, dim=0):
            out = self.forward(self._rectify_(x), training=False)
            pred_gradients = gradient(out, x).detach()
            gradients.append(pred_gradients)
            torch.cuda.empty_cache()
        gradients = torch.cat(gradients, dim=0)
        grid_img = grads2img(gradients)
        return grid_img, gradients

    @property
    def laplace(self, chunk_row=64):
        # if coord is None:
        torch.cuda.empty_cache()
        coord = nn.Parameter(self.coord)
        laplace = []
        for x in torch.split(coord, chunk_row, dim=0):
            out = self.forward(self._rectify_(x), training=False)
            pred_laplace = laplace(out, x).detach()
            laplace.append(pred_laplace)
            # del out, pred_laplace
            # gc.collect()
            torch.cuda.empty_cache()
        laplace = torch.cat(laplace, dim=0)
        laplace_img = laplace2img(laplace)
        return laplace_img, laplace

    def TV_loss(self):
        if self.feat_mode == "pe":
            return torch.tensor(0.0)
        else:
            return self.feat.TV_loss()
