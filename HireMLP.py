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


class HireMLP_Block(nn.Module):  #input shape: n, c, h, w, d
   
    def __init__(self, num_channels, region_size, shift_step, lrelu_slope=0.2, channels_reduction=4, use_bias=True, use_checkpoint=False):
        super().__init__()

        self.mlpLayer_0 = HireMlpLayer(num_channels, region_size, 0, use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.mlpLayer_1 = HireMlpLayer(num_channels, region_size, shift_step, use_bias=use_bias, use_checkpoint=use_checkpoint)

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


class HireMlpLayer(nn.Module):   #input shape: n, h, w, d, c
    
    def __init__(self, num_channels, region_size, shift_step, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.HireMLP = HireMLP(num_channels, region_size, shift_step, use_bias=use_bias)
    
    def forward_run(self, x_in):

        x = self.LayerNorm(x_in)
        x = self.HireMLP(x)
        x_out = x + x_in
        return x_out

    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class HireMLP(nn.Module): #input shape: n, h, w, d, c
    
    def __init__(self, num_channels, region_size, shift_step, use_bias=True):
        super().__init__()
        
        self.hire_h = HireUnit(num_channels, region_size, shift_step, dim=1, use_bias=use_bias)
        self.hire_w = HireUnit(num_channels, region_size, shift_step, dim=2, use_bias=use_bias)
        self.hire_d = HireUnit(num_channels, region_size, shift_step, dim=3, use_bias=use_bias)
        self.mlp_c = nn.Linear(num_channels, num_channels, bias=use_bias)

        self.reweight = MLP(num_channels, num_channels // 4, num_channels * 4)
        self.proj = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        
        B, H, W, D, C = x.shape
        
        h = self.hire_h(x)
        w = self.hire_w(x)
        d = self.hire_d(x)
        c = self.mlp_c(x)

        a = (h + w + d + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 4).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + d * a[2] + c * a[3]
        x = self.proj(x)

        return x


class HireUnit(nn.Module): #input shape: n, h, w, d, c

    def __init__(self, num_channels, region_size, shift_step, dim, use_bias=True):
        super().__init__()
        self.region_size = region_size
        self.shift_step = shift_step
        self.dim = dim
        
        self.mlp_1 = nn.Linear(num_channels*region_size, num_channels//2, bias=use_bias)
        self.LayerNorm_1 = nn.LayerNorm(num_channels//2)
        self.mlp_2 = nn.Linear(num_channels//2, num_channels*region_size, bias=use_bias)
        self.act = nn.GELU()
        
    def forward(self, x):

        B, H, W, D, C = x.shape

        # Region rearrange
        if self.shift_step>0:
            x = torch.roll(x, self.shift_step, dims=self.dim)
            
        if self.dim==1:
            pad_h = (self.region_size - H % self.region_size) % self.region_size
            if pad_h >0:
                x = nnf.pad(x, (0, 0, 0, 0, 0, 0, 0, pad_h))

            x = x.reshape(B, (H+pad_h)//self.region_size, self.region_size, W, D, C)
            x = x.permute(0, 1, 3, 4, 5, 2)
            x = x.reshape(B, (H+pad_h)//self.region_size, W, D, self.region_size*C)
        
        elif self.dim==2:
            pad_w = (self.region_size - W % self.region_size) % self.region_size
            if pad_w >0:
                x = nnf.pad(x, (0, 0, 0, 0, 0, pad_w))

            x = x.reshape(B, H, (W+pad_w)//self.region_size, self.region_size, D, C)
            x = x.permute(0, 1, 2, 4, 5, 3)
            x = x.reshape(B, H, (W+pad_w)//self.region_size, D, self.region_size*C)
            
        else: #self.dim==3:
            pad_d = (self.region_size - D % self.region_size) % self.region_size
            if pad_d >0:
                x = nnf.pad(x, (0, 0, 0, pad_d))

            x = x.reshape(B, H, W, (D+pad_d)//self.region_size, self.region_size, C)
            x = x.permute(0, 1, 2, 3, 5, 4)
            x = x.reshape(B, H, W, (D+pad_d)//self.region_size, self.region_size*C)

        # MLP
        x = self.mlp_1(x)
        x = self.LayerNorm_1(x)
        x = self.act(x)
        x = self.mlp_2(x)

        #Region restore
        if self.dim==1:
            x = x.reshape(B, (H+pad_h)//self.region_size, W, D, C, self.region_size)
            x = x.permute(0, 1, 5, 2, 3, 4)
            x = x.reshape(B, H+pad_h, W, D, C)
            x= x[:,:H].contiguous()
        
        elif self.dim==2:
            x = x.reshape(B, H, (W+pad_w)//self.region_size, D, C, self.region_size)
            x = x.permute(0, 1, 2, 5, 3, 4)
            x = x.reshape(B, H, W+pad_w, D, C)
            x= x[:,:,:W].contiguous()
            
        else: #self.dim==3:
            x = x.reshape(B, H, W, (D+pad_d)//self.region_size, C, self.region_size)
            x = x.permute(0, 1, 2, 3, 5, 4)
            x = x.reshape(B, H, W, D+pad_d, C)
            x= x[:,:,:,:D].contiguous()

        if self.shift_step>0:
            x = torch.roll(x, -self.shift_step, dims=self.dim)
        
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

    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
