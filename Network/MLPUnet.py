import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
    
    
class MLPUnet(nn.Module):
    
    def __init__(self, in_channels=1, channel_num=24, out_channels=4, use_checkpoint=True):
        super().__init__()

        self.Conv_embedding = nn.Conv3d(in_channels, channel_num, kernel_size=3, stride=1, padding='same')
        
        self.Encoder_1 = MLP_Block(channel_num, block_size=(8,8,8), grid_size=(8,8,8), use_checkpoint=use_checkpoint)
        self.Encoder_2 = MLP_Block(channel_num*2, block_size=(8,8,8), grid_size=(8,8,8), use_checkpoint=use_checkpoint)
        self.Encoder_3 = MLP_Block(channel_num*4, block_size=(6,6,6), grid_size=(6,6,6), use_checkpoint=use_checkpoint)
        self.Encoder_4 = MLP_Block(channel_num*8, block_size=(6,6,6), grid_size=(6,6,6), use_checkpoint=use_checkpoint)
        self.Encoder_5 = MLP_Block(channel_num*16, block_size=(4,4,4), grid_size=(4,4,4), use_checkpoint=use_checkpoint)

        self.downsample_1 = PatchMerging_block(channel_num)
        self.downsample_2 = PatchMerging_block(channel_num*2)
        self.downsample_3 = PatchMerging_block(channel_num*4)
        self.downsample_4 = PatchMerging_block(channel_num*8)

        self.ResBlock_1 = ResConvBlock(in_channels=channel_num, out_channels=channel_num, kernel_size=3)
        self.ResBlock_2 = ResConvBlock(in_channels=channel_num*2, out_channels=channel_num*2, kernel_size=3)
        self.ResBlock_3 = ResConvBlock(in_channels=channel_num*4, out_channels=channel_num*4, kernel_size=3)
        self.ResBlock_4 = ResConvBlock(in_channels=channel_num*8, out_channels=channel_num*8, kernel_size=3)
        self.ResBlock_5 = ResConvBlock(in_channels=channel_num*16, out_channels=channel_num*16, kernel_size=3)
        
        
        self.decoder1 = UpConvBlock(in_channels=channel_num*16,
                                    out_channels=channel_num*8,
                                    kernel_size=3,
                                    upsample_kernel_size=2)
        self.decoder2 = UpConvBlock(in_channels=channel_num*8,
                                    out_channels=channel_num*4,
                                    kernel_size=3,
                                    upsample_kernel_size=2)
        self.decoder3 = UpConvBlock(in_channels=channel_num*4,
                                    out_channels=channel_num*2,
                                    kernel_size=3,
                                    upsample_kernel_size=2)
        self.decoder4 = UpConvBlock(in_channels=channel_num*2,
                                    out_channels=channel_num,
                                    kernel_size=3,
                                    upsample_kernel_size=2)
        
        self.Conv = nn.Conv3d(channel_num, out_channels, kernel_size=3, stride=1, padding='same')
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x_in):

        # Swin-Trans
        x = self.Conv_embedding(x_in)
        x_mlp_1 = self.Encoder_1(x)
        
        x = self.downsample_1(x_mlp_1)
        x_mlp_2 = self.Encoder_2(x)
        
        x = self.downsample_2(x_mlp_2)
        x_mlp_3 = self.Encoder_3(x)

        x = self.downsample_3(x_mlp_3)
        x_mlp_4 = self.Encoder_4(x)

        x = self.downsample_4(x_mlp_4)
        x_mlp_5 = self.Encoder_5(x)

        # Residual block
        x_1 = self.ResBlock_1(x_mlp_1)
        x_2 = self.ResBlock_2(x_mlp_2)
        x_3 = self.ResBlock_3(x_mlp_3)
        x_4 = self.ResBlock_4(x_mlp_4)
        x_5 = self.ResBlock_5(x_mlp_5)

        # Upsampling decoder
        x = self.decoder1(x_5, x_4)
        x = self.decoder2(x, x_3)
        x = self.decoder3(x, x_2)
        x = self.decoder4(x, x_1)

        # Segmentation head
        x = self.Conv(x)
        Seg_pred = self.Softmax(x)
        return Seg_pred
    

########################################################
# Blocks
########################################################

class MLP_Block(nn.Module):  #input shape: n, c, h, w,œ d
   
    def __init__(self, num_channels, block_size, grid_size, lrelu_slope=0.2,
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, 
                 channels_reduction=4, dropout_rate=0.0, use_bias=True, use_checkpoint=False):
        super().__init__()
        
        self.mlpLayer_0 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=block_size, grid_size=grid_size, 
                                                              num_channels=num_channels, input_proj_factor=input_proj_factor,
                                                              block_gmlp_factor=block_gmlp_factor, grid_gmlp_factor=grid_gmlp_factor, 
                                                              dropout_rate=dropout_rate, use_bias=use_bias, use_checkpoint=use_checkpoint)
        self.channel_attention_block_0 = RCAB(num_channels=num_channels, reduction=channels_reduction, lrelu_slope=lrelu_slope, 
                                              use_bias=use_bias, use_checkpoint=use_checkpoint)
        
        self.mlpLayer_1 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=block_size, grid_size=grid_size, 
                                                              num_channels=num_channels, input_proj_factor=input_proj_factor,
                                                              block_gmlp_factor=block_gmlp_factor, grid_gmlp_factor=grid_gmlp_factor, 
                                                              dropout_rate=dropout_rate, use_bias=use_bias, use_checkpoint=use_checkpoint)
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


class ResConvBlock(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int):
        super().__init__()
        
        self.ReLU = nn.ReLU()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same')
        
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding='same')
            self.norm3 = nn.BatchNorm3d(out_channels)

    def forward(self, x_in):
        
        x = self.conv1(x_in)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x_conv = self.norm2(x)
        
        if hasattr(self, "conv3"):
            x = self.conv3(x_in)
            x_res = self.norm3(x)
        else:
            x_res = x_in
            
        x = x_conv + x_res
        x_out = self.ReLU(x)
        
        return x_out

    
class UpConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 upsample_kernel_size: int):
        super().__init__()
        
        self.ReLU = nn.ReLU()
        self.trans_conv = nn.ConvTranspose3d(in_channels, out_channels, 
                                             kernel_size=upsample_kernel_size, 
                                             stride=upsample_kernel_size)
        self.conv_block = ResConvBlock(in_channels=out_channels*2, 
                                       out_channels=out_channels, 
                                       kernel_size=kernel_size)

    def forward(self, x_up, x_skip):
        
        x = self.trans_conv(x_up)
        x = torch.cat((x, x_skip), dim=1)
        x_out = self.conv_block(x)
        
        return x_out


class PatchMerging_block(nn.Module):

    def __init__(self, embed_dim: int):

        super().__init__()
        
        self.down_conv = nn.Conv3d(embed_dim, embed_dim*2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim*2)

    def forward(self, x):

        b, c, d, h, w = x.shape
        if (d % 2 == 1) or (h % 2 == 1) or (w % 2 == 1):
            x = nnf.pad(x, (0, w % 2, 0, h % 2, 0, d % 2))
        
        x = self.down_conv(x)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')
        
        return x_out
    

class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):   #input shape: n, h, w, d, c
    """The multi-axis gated MLP block."""
    
    def __init__(self, block_size, grid_size, num_channels, 
                 input_proj_factor=2, block_gmlp_factor=2, grid_gmlp_factor=2, use_bias=True, dropout_rate=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels*input_proj_factor, bias=use_bias)
        self.gelu = nn.GELU()
        self.GridGmlpLayer = GridGmlpLayer(grid_size=grid_size, num_channels=num_channels*input_proj_factor//2, 
                                           use_bias=use_bias, factor=grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=block_size, num_channels=num_channels*input_proj_factor//2, 
                                             use_bias=use_bias, factor=block_gmlp_factor)
        self.out_project = nn.Linear(num_channels*input_proj_factor, num_channels, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward_run(self, x_in):
        
        x = self.LayerNorm(x_in)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)
        
        #grid gMLP
        u = self.GridGmlpLayer(u)
        #block gMLP
        v = self.BlockGmlpLayer(v)
        
        #out projection
        x = torch.cat([u,v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        
        x_out = x + x_in
        return x_out

    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)
        return x_out


class GridGmlpLayer(nn.Module):  #input shape: n, h, w, d, c
    """Grid gMLP layer that performs global mixing of tokens."""
    
    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.gh = grid_size[0]
        self.gw = grid_size[1]
        self.gd = grid_size[2]
        
        self.LayerNorm = nn.LayerNorm(num_channels)
        self.in_project = nn.Linear(num_channels, num_channels*factor, use_bias)  #c->c*factor
        self.gelu = nn.GELU()
        self.GridGatingUnit = GridGatingUnit(num_channels*factor, n=self.gh*self.gw*self.gd)  #c*factor->c*factor//2
        self.out_project = nn.Linear(num_channels*factor//2, num_channels, use_bias)   #c*factor//2->c
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        
        _, h, w, d, _ = x.shape
        
        # padding
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.gh - h % self.gh) % self.gh
        pad_b = (self.gw - w % self.gw) % self.gw
        pad_r = (self.gd - d % self.gd) % self.gd
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

        fh, fw, fd = x.shape[1] // self.gh, x.shape[2] // self.gw, x.shape[3] // self.gd
        x = block_images_einops(x, patch_size=(fh, fw, fd))  #n (gh gw gd) (fh fw fd) c
        
        # gMLP: Global (grid) mixing part, provides global grid communication.
        shortcut = x
        x = self.LayerNorm(x)
        x = self.in_project(x)
        x = self.gelu(x)
        x = self.GridGatingUnit(x)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut
        
        x = unblock_images_einops(x, grid_size=(self.gh, self.gw, self.gd), patch_size=(fh, fw, fd))
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()
        
        return x


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


class GridGatingUnit(nn.Module):  #input shape: n (gh gw gd) (fh fw fd) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
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
        v = v.permute(0, 3, 2, 1)  #n, c/2, (fh fw fd) (gh gw gd)
        v = self.Dense_0(v)
        v = v.permute(0, 3, 2, 1)  #n (gh gw gd) (fh fw fd) c/2
        
        return u*(v + 1.0)


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
    
