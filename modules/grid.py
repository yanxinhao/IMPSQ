# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2022-05-16 08:20:17
LastEditors: yanxinhao
FilePath: /ImpSq/modules/grid.py
Date: 2022-05-16 08:20:17
Description: 
"""
# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2022-05-14 12:39:27
LastEditors: yanxinhao
FilePath: /PhasorImage/grid.py
Date: 2022-05-14 12:34:01
Description: 
"""
import torch
import torch.nn as nn
from .image_util import grid_sample
import torch.nn.functional as F


class Grid(nn.Module):
    def __init__(self, dim_feat, gridSize, device="cuda"):
        super(Grid, self).__init__()
        self.dim_feat = dim_feat
        self.gridsize = gridSize
        self.device = device
        self.init_()

    def init_(self):
        # self.params = nn.Parameter(torch.zeros([1, self.dim_feat] + self.gridsize)).to(
        #     self.device
        # )
        self.params = nn.Parameter(
            torch.zeros([1, self.dim_feat] + self.gridsize, device=self.device)
        )

    def TV_loss(self, weight=0.0):
        tv_x = self.params.diff(dim=2).abs().mean()
        tv_y = self.params.diff(dim=3).abs().mean()
        loss = weight * (tv_x + tv_y)
        return loss

    def forward(self, coord):
        """sample in grid

        Args:
            coord (tensor): [n,2]

        Returns:
            feature: []
        """
        feat = (
            grid_sample(self.params, coord[None, :, None, :])
            .squeeze(0)
            .squeeze(-1)
            .transpose(0, 1)
        )
        return feat

