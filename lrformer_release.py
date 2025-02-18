from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.modules.utils import _pair as to_2tuple


import numpy as np


from mmseg.models.builder import BACKBONES
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.logging import print_log

from mmcv.cnn import get_model_complexity_info

from mmcv.cnn import build_norm_layer


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.GELU, extra_act=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.act1 = act_layer() if extra_act else nn.Identity()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0,2,1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0,2,1)


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pooled_sizes=[11,8,6,4], q_pooled_size=1, q_conv=False):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001
        
        self.norm = nn.LayerNorm(dim)
        
        self.q_pooled_size = q_pooled_size
        
        # Useless code
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        H, W = int(H), int(W)
        
        if self.q_pooled_size > 1:
            # Too keep the W/H ratio of the features
            q_pooled_size = (self.q_pooled_size, round(W*float(self.q_pooled_size)/H + self.eps)) \
                if W >= H else (round(H*float(self.q_pooled_size)/W + self.eps), self.q_pooled_size)
            
            # Conduct fixed pooled size pooling on q
            q = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(B, C, H, W), q_pooled_size)
            _, _, H1, W1 = q.shape
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            H1, W1 = H, W
            if self.q_conv is not None:
                x1 = x.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1)
                q = x1 + self.q_conv(x1)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        # Conduct Pyramid Pooling on K, V
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pooled_size, l) in zip(self.pooled_sizes, d_convs):
            pooled_size = (pooled_size, round(W*pooled_size/H + self.eps)) if W >= H else (round(H*pooled_size/W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x_, pooled_size)
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   # B N C
        x = x.transpose(1,2).reshape(B, -1, C)
        
        x = self.proj(x)
        
        # Bilinear upsampling for residual connection
        if self.q_pooled_size > 1:
            x = x.transpose(1, 2).reshape(B, C, H1, W1)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x.view(B, C, -1).transpose(1, 2)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ls=False, pooled_sizes=[12,16,20,24], q_pooled_size=1, q_conv=False, extra_act=False, use_prenorm=False):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(dim) if use_prenorm else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, pooled_sizes=pooled_sizes, q_pooled_size=q_pooled_size, q_conv=q_conv)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3, extra_act=extra_act)
        
        self.ls = ls # layer scale
        if self.ls:
            layer_scale_init_value = 1e-6
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        # update: removed dwconvs
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    
    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape
        
        #print("block init x.max", x.max())
        x = x.permute(0,2,1).reshape(B, C, H, W)
        x = self.pre_norm(x)
        x = self.cpe(x) + x
        x = x.reshape(B, C, -1).permute(0,2,1)
        #print("block after cpe x.max", x.max())
        
        if self.ls:
            x = x + self.drop_path(self.layer_scale_1[None, None, :] * self.attn(self.norm1(x), H, W, d_convs=d_convs))
            x = x + self.drop_path(self.layer_scale_2[None, None, :] * self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W, d_convs=d_convs))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=kernel_size//2)
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=64, patch_size=4):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_chans, out_chans//2, 3, 2, 1),
                nn.BatchNorm2d(out_chans//2),
                nn.GELU(),
                nn.Conv2d(out_chans//2, out_chans//2, 3, 1, 1),
                nn.BatchNorm2d(out_chans//2),
                nn.GELU(),
                nn.Conv2d(out_chans//2, out_chans, 3, 2, 1),
                nn.BatchNorm2d(out_chans),
                nn.GELU(),
                nn.Conv2d(out_chans, out_chans, 3, 1, 1),
        )
        self.norm = nn.LayerNorm(out_chans, eps=1e-6)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
    
    def forward(self, x):
        x = self.conv1(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, (H, W)


@BACKBONES.register_module()
class LRFormer(BaseModule):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), q_pooled_sizes=[16,16,16,16],
                 depths=[3, 3, 12, 3], q_conv=False, use_ls=True, extra_act=True, init_cfg=None, **kwargs): #
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims
        
        # print(q_pooled_sizes, depths, "q_conv:", q_conv, "use_ls", use_ls)
        
        # Original 11^2 + 8^2 + 6^2 + 4^2 = 237 < 16^2
        # Target Pooled Feature Size for Pyramid Pooling
        # This is NOT the pooling ratio
        pooled_sizes = [[11, 8, 6, 4], [11, 8, 6, 4], [11, 8, 6, 4], [11, 8, 6, 4]]

        self.patch_embed1 = Stem(in_chans=in_chans, out_chans=embed_dims[0], patch_size=4)

        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                    embed_dim=embed_dims[1], overlap=True)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                    embed_dim=embed_dims[2], overlap=True)
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                    embed_dim=embed_dims[3], overlap=True)
        
        self.d_convs1 = nn.ModuleList([nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pooled_sizes[0]])
        self.d_convs2 = nn.ModuleList([nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pooled_sizes[1]])
        self.d_convs3 = nn.ModuleList([nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pooled_sizes[2]])
        self.d_convs4 = nn.ModuleList([nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pooled_sizes[3]])
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0


        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=False,norm_layer=norm_layer, pooled_sizes=pooled_sizes[0], 
            q_pooled_size=q_pooled_sizes[0], q_conv=q_conv, extra_act=extra_act)
            for i in range(depths[0])])
        

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=False, norm_layer=norm_layer, pooled_sizes=pooled_sizes[1], 
            q_pooled_size=q_pooled_sizes[1],q_conv=q_conv, extra_act=extra_act)
            for i in range(depths[1])])

        cur += depths[1]

        
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=use_ls, norm_layer=norm_layer, pooled_sizes=pooled_sizes[2], 
            q_pooled_size=q_pooled_sizes[2], q_conv=q_conv, extra_act=extra_act, use_prenorm=i in [7,15] if depths[2] > 20 else 0) # prenorm to avoid fp16 overflow
            for i in range(depths[2])])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], ls=use_ls, norm_layer=norm_layer, pooled_sizes=pooled_sizes[3],
            q_pooled_size=q_pooled_sizes[3],q_conv=q_conv, extra_act=extra_act)
            for i in range(depths[3])])
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)


    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                if table_key in self.state_dict():
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    L2, nH2 = table_current.size()
                    if nH1 != nH2:
                        print_log(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(
                                1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv2d):
        #    nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #    if isinstance(m, nn.Conv2d) and m.bias is not None:
        #        nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        B = x.shape[0]
        
        out = []

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        
        for idx, blk in enumerate(self.block1):
            x = blk(x, H, W, self.d_convs1)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)

        for idx, blk in enumerate(self.block2):
            x = blk(x, H, W, self.d_convs2)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out.append(x)

        # stage 3
        x, (H, W) = self.patch_embed3(x)

        for idx, blk in enumerate(self.block3):
            x = blk(x, H, W, self.d_convs3)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out.append(x)
        
        # stage 4
        x, (H, W) = self.patch_embed4(x)

        for idx, blk in enumerate(self.block4):
            x = blk(x, H, W, self.d_convs4)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out.append(x)
        
        return out
    
    def forward(self, x):
        x = self.forward_features(x)
        
        return x
    
    def forward_for_fpn(self, x):
        return self.forward_features_for_fpn(x)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

