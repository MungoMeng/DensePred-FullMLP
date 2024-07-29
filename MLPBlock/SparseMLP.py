import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal


class SparseMLP_Block(nn.Module):  #input shape: n, c, h, w, d
   
    def __init__(self, num_channels, feat_size, lrelu_slope=0.2, channels_reduction=4, use_bias=True, use_checkpoint=False):
        super().__init__()

        self.mlpLayer_0 = SparseMlpLayer(num_channels, feat_size, use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.mlpLayer_1 = SparseMlpLayer(num_channels, feat_size, use_bias=use_bias, use_checkpoint=use_checkpoint)

        self.channel_attention_block_0 = RCAB(num_channels=num_channels, reduction=channels_reduction, lrelu_slope=lrelu_slope,
                                              use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.channel_attention_block_1 = RCAB(num_channels=num_channels, reduction=channels_reduction, lrelu_slope=lrelu_slope,
                                              use_bias=use_bias, use_checkpoint=use_checkpoint)
    
    def forward(self, x_in):

        x = x_in.permute(0,2,3,4,1)  #n,h,w,d,c
        x = self.mlpLayer_0(x)
        x = self.channel_attention_block_0(x)
        x = self.mlpLayer_1(x)
        x = self.channel_attention_block_1(x)
        x = x.permute(0,4,1,2,3)  #n,c,h,w,d
        
        x_out = x + x_in
        return x_out


class SparseMlpLayer(nn.Module):   #input shape: n, h, w, d, c
    
    def __init__(self, num_channels, feat_size, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.DWConv = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding='same', groups=num_channels, bias=use_bias)
        self.norm = nn.LayerNorm(num_channels)
        self.activation = nn.GELU()
        self.SpareMLP = SpareMLP(num_channels, feat_size, use_bias=use_bias)
    
    def forward_run(self, x_in):

        x = self.DWConv(x_in.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = self.activation(self.norm(x))
        x = self.SpareMLP(x)
        
        x_out = x + x_in
        return x_out

    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class SpareMLP(nn.Module): #input shape: n, h, w, d, c
    
    def __init__(self, num_channels, feat_size, use_bias=True):
        super().__init__()
        
        self.norm = nn.LayerNorm(num_channels)
        self.activation = nn.GELU()
        
        self.proj_h = nn.Linear(feat_size[0], feat_size[0], bias=use_bias)
        self.proj_w = nn.Linear(feat_size[1], feat_size[1], bias=use_bias)
        self.proj_d = nn.Linear(feat_size[2], feat_size[2], bias=use_bias)
        self.fuse = nn.Linear(num_channels*4, num_channels, bias=use_bias)

    def forward(self, x):

        x = self.activation(self.norm(x))
        x_h = self.proj_h(x.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
        x_w = self.proj_w(x.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
        x_d = self.proj_d(x.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        x = self.fuse(torch.cat([x, x_h, x_w, x_d], dim=-1))
        
        return x


class RCAB(nn.Module):  #input shape: n, h, w, d, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
    
    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction)
    
    def forward_run(self, x):
        
        shortcut = x
        x = self.LayerNorm(x)
        
        x = x.permute(0,4,1,2,3)  #n,c,h,w,d
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0,2,3,4,1)  #n,h,w,d,c
        
        x = self.channel_attention(x)
        x_out = x + shortcut
        
        return x_out

    def forward(self, x):
        
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x)
        else:
            x = self.forward_run(x)
        return x


class CALayer(nn.Module):  #input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""
    
    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()
        
        self.Conv_0 = nn.Conv3d(num_channels, num_channels//reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv3d(num_channels//reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_in):
        
        x = x_in.permute(0,4,1,2,3)  #n,c,h,w,d
        x = torch.mean(x, dim=(2,3,4), keepdim=True)
        x = self.Conv_0(x)
        x = self.relu(x)
        x = self.Conv_1(x)
        w = self.sigmoid(x)
        w = w.permute(0,2,3,4,1)  #n,h,w,d,c

        x_out = x_in*w
        return x_out
