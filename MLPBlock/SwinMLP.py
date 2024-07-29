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


class SwinMLP_Block(nn.Module):  #input shape: n, c, h, w, d
   
    def __init__(self, num_channels, block_size, shift_size, lrelu_slope=0.2, channels_reduction=4, use_bias=True, use_checkpoint=False):
        super().__init__()
        
        self.mlpLayer_0 = SwinMlpLayer(num_channels, block_size, shift_size, use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.channel_attention_block_0 = RCAB(num_channels=num_channels, reduction=channels_reduction, lrelu_slope=lrelu_slope,
                                              use_bias=use_bias, use_checkpoint=use_checkpoint)
        
        self.mlpLayer_1 = SwinMlpLayer(num_channels, block_size, shift_size, use_bias=use_bias, use_checkpoint=use_checkpoint)
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


class SwinMlpLayer(nn.Module):   #input shape: n, h, w, d, c
    
    def __init__(self, num_channels, block_size, shift_size, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.shift_size = shift_size
        
        self.BlockGmlpLayer_1 = BlockGmlpLayer(block_size=block_size, num_channels=num_channels, use_bias=use_bias)
        self.BlockGmlpLayer_2 = BlockGmlpLayer(block_size=block_size, num_channels=num_channels, use_bias=use_bias)
    
    def forward_run(self, x_in):

        #block gMLP
        x = self.BlockGmlpLayer_1(x_in)

        #shifted block gMLP
        x = nnf.pad(x, (0, 0, self.shift_size[2], 0, self.shift_size[1], 0, self.shift_size[0], 0) , "constant", 0)
        x = self.BlockGmlpLayer_2(x)
        x = x[:,self.shift_size[0]:,self.shift_size[1]:,self.shift_size[2]:,:]
        
        x_out = x + x_in
        return x_out

    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class BlockGmlpLayer(nn.Module):  #input shape: n, h, w, d, c
    """Block gMLP layer that performs local mixing of tokens."""
    
    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0.0):
        super().__init__()

        self.fh = block_size[0]
        self.fw = block_size[1]
        self.fd = block_size[2]
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels*factor, use_bias)   #c->c*factor
        self.gelu = nn.GELU()
        self.BlockGatingUnit = BlockGatingUnit(num_channels*factor, n=self.fh*self.fw*self.fd)   #c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels*factor//2, num_channels, use_bias)   #c*factor//2->c
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        
        _, h, w, d, _ = x.shape

        # padding
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.fh - h % self.fh) % self.fh
        pad_b = (self.fw - w % self.fw) % self.fw
        pad_r = (self.fd - d % self.fd) % self.fd
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        
        gh, gw, gd = x.shape[1] // self.fh, x.shape[2] // self.fw, x.shape[3] // self.fd
        x = block_images_einops(x, patch_size=(self.fh, self.fw, self.fd))  #n (gh gw gd) (fh fw fd) c
        
        # gMLP: Local (block) mixing part, provides local block communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.BlockGatingUnit(x)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut
        
        x = unblock_images_einops(x, grid_size=(gh, gw, gd), patch_size=(self.fh, self.fw, self.fd))
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        
        return x


class BlockGatingUnit(nn.Module):  #input shape: n (gh gw gd) (fh fw fd) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the third.
    """
    def __init__(self, c, n, use_bias=True):
        super().__init__()
        
        self.Dense_0 = nn.Linear(n, n, use_bias)
        self.LayerNorm = nn.LayerNorm(c//2)
        
    def forward(self, x):
        
        c = x.size(-1)
        c = c // 2
        u, v  = torch.split(x, c, dim=-1)
        
        v = self.LayerNorm(v)
        v = v.permute(0, 1, 3, 2)  #n, (gh gw gd), c/2, (fh fw fd)
        v = self.Dense_0(v)
        v = v.permute(0, 1, 3, 2)  #n (gh gw gd) (fh fw fd) c/2
        
        return u*(v + 1.0)


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

    
########################################################
# Functions
########################################################

def block_images_einops(x, patch_size):  #n, h, w, d, c
    """Image to patches."""
    
    batch, height, width, depth, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    grid_depth = depth // patch_size[2]
    
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c",
        gh=grid_height, gw=grid_width, gd=grid_depth, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    
    x = einops.rearrange(
        x, "n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c",
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x
    
