import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.norm import build_norm_layer

from mmseg.models.utils import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.GELU, extra_act=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        return x


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
        pooled_sizes=[1,2,3,6], q_pooled_size=-1, **kwargs):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001
        
        self.norm = nn.LayerNorm(dim)
        
        self.q_pooled_size = q_pooled_size
        self.upsample = nn.Upsample(scale_factor=q_pooled_size) if q_pooled_size > 1 else nn.Identity()
        
    def forward(self, x, d_convs=None):
        B, C, H, W = x.shape
        H, W = int(H), int(W)

        if d_convs is None:
            d_convs = [None] * len(self.pooled_sizes)
        
        if self.q_pooled_size > -1:
            q_pooled_size = self.q_pooled_size
            q_pooled_size = (q_pooled_size, round(W*float(q_pooled_size)/H + self.eps)) if W >= H else (round(H*float(q_pooled_size)/W + self.eps), q_pooled_size)
            q = F.adaptive_avg_pool2d(x, q_pooled_size)
            _, _, H1, W1 = q.shape
            q = self.q(q).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2).contiguous() # B, N, C
        else:
            q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2).contiguous() # B, N, C
        
        pools = []
        for (pooled_size, l) in zip(self.pooled_sizes, d_convs):
            pooled_size = (pooled_size, round(W*pooled_size/H + self.eps)) if W >= H else (round(H*pooled_size/W + self.eps), pooled_size)
            pool = F.adaptive_avg_pool2d(x, pooled_size)
            if l is not None:
                pool = pool + l(pool) # fix backward bug in higher torch versions when training
            pools.append(pool.view(B, C, -1))
        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1)) 
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale # B Head N N
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   # B Head N C
        x = x.transpose(1,2).reshape(B, -1, C)
        
        # post conv seem not good: 83.4, id 9677
        #if self.q_conv is not None:
        #    qpe = self.q_conv(q.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1))
        #    x = qpe.view(B, C, -1).transpose(1,2) + x
        
        x = self.proj(x)
        
        if self.q_pooled_size > -1:
            x = x.transpose(1, 2).reshape(B, C, H1, W1)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        else:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x




class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_cfg=None, ls=True, pooled_sizes=[12,16,20,24], q_pooled_size=1, q_conv=False, extra_act=False):
        super().__init__()
        _, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = PoolingAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, pooled_sizes=pooled_sizes, q_pooled_size=q_pooled_size)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3, extra_act=extra_act)        
        
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.ls = ls # layer scale
        if self.ls:
            layer_scale_init_value = 1e-6
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
    
    def forward(self, x, d_convs=None):
        x = self.cpe(x) + x
        
        if self.ls:
            x = x + self.drop_path(self.layer_scale_1[None, :, None, None] * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2[None, :, None, None] * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


@HEADS.register_module()
class Pool2Head(BaseDecodeHead):
    """Pooling Attention Head.
    Args:
        num_heads: number of head in the pooling attention
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        q_pooled_size: the size of the query in the pooling attention
    """

    def __init__(self,
                 num_heads=8,
                 mlp_ratio=4,
                 q_pooled_size=32,
                 qkv_bias=True,
                 **kwargs):
        super(Pool2Head, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.squeeze1 = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.squeeze2 = ConvModule(
            self.channels + self.in_channels[-1],
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        base_pooled_sizes = [11, 8, 6, 4] # basic ratio corresponding q_pooled_size=16
        pooled_sizes = [round(temp *  q_pooled_size / 16. + 0.001) for temp in base_pooled_sizes]
        
        self.fuse_block1 = Block(self.channels, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None, drop=0, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_cfg=self.norm_cfg, pooled_sizes=pooled_sizes, q_pooled_size=q_pooled_size)
        self.fuse_block2 = Block(self.channels, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None, drop=0, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_cfg=self.norm_cfg, pooled_sizes=pooled_sizes, q_pooled_size=q_pooled_size)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs] # concat
        
        inputs_last = inputs[-1]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze1(inputs) # squeeze1
        
        x = self.fuse_block1(x) # fuse1
        
        x = torch.cat((x, inputs_last), dim=1)
        x = self.squeeze2(x) # squeeze2
        
        x = self.fuse_block2(x) # fuse2

        output = self.cls_seg(x)
        return output