# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, CA_Block
import math
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_encoder
from einops import rearrange
from torch.distributions import Categorical, Normal
# import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

class MultiAgentAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone, 默认vit-large
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, add_critic = False):
        super().__init__()
        # parameter
        self.n_agents = 3
        self.q_flag = False
        self.grid_size = 8
        self.map_real_w = 125
        self.map_real_h = 125
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        self.max_speed = 8
        self.max_theta = math.pi/3
        self.state_scale = torch.tensor([self.map_real_w, self.map_real_h, self.max_theta])
        self.dw = self.map_real_w/self.grid_size/2
        self.dh = self.map_real_h/self.grid_size/2
        self.add_critic = add_critic
        # model(encoder)
        self.embed_dim = 128
        drop_rate = 0.
        norm_layer=nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=128, patch_size=16, in_chans=3, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.action_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.state_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.ca_blocks = nn.ModuleList([
            CA_Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.norm = norm_layer(self.embed_dim)
        self.to_local_region = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.to_local_point  = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self.others_norm = norm_layer(self.embed_dim)
        self.to_local_state = nn.Linear(self.embed_dim, 2)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)
        self.mask_conv.weight.data.fill_(1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        if self.add_critic:
            self.critic_blocks = nn.ModuleList([
                Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(depth-2)])
            self.critic_norm = norm_layer(self.embed_dim)
            self.to_value = nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1))
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_encoder(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed_encoder(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.action_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight) # 均匀分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # 常数为0初始化
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs): # 图片划分为块
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x): # 块返回为图片
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # 需要保留块的数目
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio = None, nei = None):
        self.B = x.shape[0]
        if nei is None:
            # calculate Invalid Action Mask 
            agent_boundary_map = x[:,0,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            # encoder
            x = rearrange(x, 'B n c h w -> (B n) c h w')    
            x = self.patch_embed(x)
        else:
            agent_boundary_map = x[:,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            x = self.patch_embed(x)
            x = torch.cat((x.unsqueeze(1), nei), dim = 1)
            x = rearrange(x, 'B n p l -> (B n) p l')
        # x = rearrange(x, '(B n) p l -> B (n p) l', B = self.B)
        action_token = self.action_token + self.pos_embed[:, :2, :]
        action_token = action_token.expand(x.shape[0], -1, -1)
        state_token = self.state_token + self.pos_embed[:, 2, :]
        state_token = state_token.expand(x.shape[0], -1, -1)
        x = torch.cat((action_token, state_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        # cross attention
        x = rearrange(x, '(B n) p l -> B n p l', B = self.B)    
        agent = x[:, 0]
        if self.add_critic:
            # For Critic
            feature = agent
            for blk in self.critic_blocks:
                feature = blk(feature)
            feature = self.critic_norm(feature)
            feature = self.to_value(feature[:,0])
        others = self.others_norm(rearrange(x[:, 1:], 'B n p l -> B (n p) l'))
        for blk in self.ca_blocks:
            agent = blk(agent, others)
        x = agent
        x = self.norm(x) # x.shape = [B d l] = [32 1178 128]
        out_local_region = self.to_local_region(x[:,0])
        out_local_region = torch.where(self.IAM_mask, out_local_region, torch.tensor(-1e4).to(dtype=out_local_region.dtype, device=out_local_region.device))
        out_local_point = self.to_local_point(x[:,1])
        state = self.to_local_state(x[:,2])
        action = torch.cat((out_local_region, out_local_point), dim =1)
        if self.add_critic:
            return x, action, state, feature
        else:
            return x, action, state, []

    def forward_decoder(self, x):
        # embed tokens
        x = x[:, :67, :]
        x = self.decoder_embed(x) # 全连接层，将encoder降为为decoder的输入

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove actions and state token
        x = x[:, 3:, :]

        return x
    
    def global_to_local(self, global_point, car_pos):
        '''
        Calculate the local position of a point in the global map
        '''
        local_x = global_point[:, 0] - car_pos[:, 0]
        local_y = global_point[:, 1] - car_pos[:, 1]

        center = torch.tensor([self.map_real_w / 2, self.map_real_h / 2])
        local_x += center[0]
        local_y += center[1]

        return torch.stack((local_x, local_y), dim=1)
    
    def local_to_global(self, local_point, car_pos):
        '''
        Calcluate the global position of the goal from local map
        '''
        map_w = 125
        map_h = 125
        center = torch.tensor([map_w / 2, map_h / 2])
        x = local_point[:, 0] - center[0]
        y = local_point[:, 1] - center[1]
        return torch.stack((car_pos[:, 0] + x, car_pos[:, 1] + y), dim=1)
    
    
    def forward_loss(self, gt, pred, state, out_state, out_action, img, value, gt_value):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(gt) # 使用预测图作为输出target是 [N, L, p*p*3]
        # target = self.patchify(img[:, 0]) # 使用原图作为输出
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        # image loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum() / loss.numel()
        # state loss
        agent_state = state[:, [0, 1, 3]]/self.state_scale.to(state.device)
        s_loss = (out_state - agent_state[:,:2]) ** 2
        s_loss = s_loss.sum()/s_loss.numel()
        # action loss
        region_dist =  torch.distributions.categorical.Categorical(logits=out_action[:,:64])
        point_dist = torch.distributions.normal.Normal(out_action[:,64:66], torch.exp(out_action[:,-2:]))
        regions = region_dist.sample()
        points = point_dist.sample()
        real_point = state[:,-2:]
        # real_point_ = real_point.clone()
        real_point = self.global_to_local(real_point, state)
        real_point[:, 0] = torch.clamp(real_point[:, 0], max=self.map_real_w-1e-5, min = 0.01)  
        real_point[:, 1] = torch.clamp(real_point[:, 1], max=self.map_real_h-1e-5, min = 0.01) 
        real_regions = real_point // torch.Tensor([self.map_real_w/self.grid_size, self.map_real_h/self.grid_size]).to(real_point.device)
        gt_regions = real_regions[:,1] + self.grid_size * real_regions[:,0]
        x = (real_regions[:,0] * 2 + 1) * self.dw  
        y = (real_regions[:,1] * 2 + 1) * self.dh
        gt_points = torch.stack(((x-real_point[:,0])/self.dw, (y-real_point[:,1])/self.dh), dim = 1)
        region_loss = -region_dist.log_prob(gt_regions).mean()
        point_loss = -point_dist.log_prob(gt_points).mean()
        result = self.calculate_result(gt_regions, regions, gt_points, points)
        # Critic loss
        if self.add_critic:
            critic_loss = F.smooth_l1_loss(value, gt_value, beta=10)
            loss =  loss + region_loss + point_loss + s_loss + critic_loss
            print(loss)
        else:
            # sum all loss
            # loss = region_loss + point_loss 
            loss = region_loss + s_loss + loss
            # loss =  region_loss + point_loss
        return loss, result  

    
    def forward(self, imgs, state, gt, gt_value, mask_ratio=0.75):
        
        latent, out_action, out_state, value = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss, result = self.forward_loss(gt, pred, state, out_state, out_action, imgs, value, gt_value)
        return loss, result
    
    def calculate_result(self, gt_regions, pre_regions, gt_points, pre_points):
        result = {}
        result['regions_true'] = torch.sum(gt_regions == pre_regions)
        result['points_loss']  = F.mse_loss(pre_points, gt_points, reduction='sum')
        return result


class MultiAgentAutoencoder_withCritic(nn.Module):
    """ 
    使用Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC 的方法
    既训练actor也训练critic
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # parameter
        self.n_agents = 3
        self.q_flag = False
        self.grid_size = 8
        self.map_real_w = 125
        self.map_real_h = 125
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        self.max_speed = 8
        self.max_theta = math.pi/3
        self.state_scale = torch.tensor([self.map_real_w, self.map_real_h, self.max_theta])
        self.dw = self.map_real_w/self.grid_size/2
        self.dh = self.map_real_h/self.grid_size/2
        # model(encoder)
        self.embed_dim = 128
        drop_rate = 0.
        norm_layer=nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=128, patch_size=16, in_chans=3, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.action_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.state_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.ca_blocks = nn.ModuleList([
            CA_Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.critic_blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.norm = norm_layer(self.embed_dim)
        self.critic_norm = norm_layer(self.embed_dim)
        self.to_local_region = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.to_local_point  = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self.others_norm = norm_layer(self.embed_dim)
        self.to_local_state = nn.Linear(self.embed_dim, 2)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)
        self.mask_conv.weight.data.fill_(1.0)
        self.to_value = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1))
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_encoder(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed_encoder(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.action_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight) # 均匀分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # 常数为0初始化
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs): # 图片划分为块
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x): # 块返回为图片
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # 需要保留块的数目
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio = None, nei = None):
        self.B = x.shape[0]
        if nei is None:
            # calculate Invalid Action Mask 
            agent_boundary_map = x[:,0,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            # encoder
            x = rearrange(x, 'B n c h w -> (B n) c h w')    
            x = self.patch_embed(x)
        else:
            agent_boundary_map = x[:,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            x = self.patch_embed(x)
            x = torch.cat((x.unsqueeze(1), nei), dim = 1)
            x = rearrange(x, 'B n p l -> (B n) p l')
        # x = rearrange(x, '(B n) p l -> B (n p) l', B = self.B)
        action_token = self.action_token + self.pos_embed[:, :2, :]
        action_token = action_token.expand(x.shape[0], -1, -1)
        state_token = self.state_token + self.pos_embed[:, 2, :]
        state_token = state_token.expand(x.shape[0], -1, -1)
        x = torch.cat((action_token, state_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        # cross attention
        x = rearrange(x, '(B n) p l -> B n p l', B = self.B)    
        agent = x[:, 0]
        # For Critic
        feature = agent
        for blk in self.critic_blocks:
            feature = blk(feature)
        feature = self.critic_norm(feature)
        feature = self.to_value(feature[:,0])
        # Continue Actor
        others = self.others_norm(rearrange(x[:, 1:], 'B n p l -> B (n p) l'))
        for blk in self.ca_blocks:
            agent = blk(agent, others)
        x = agent
        x = self.norm(x) # x.shape = [B d l] = [32 1178 128]
        out_local_region = self.to_local_region(x[:,0])
        out_local_region = torch.where(self.IAM_mask, out_local_region, torch.tensor(-1e4).to(dtype=out_local_region.dtype, device=out_local_region.device))
        out_local_point = self.to_local_point(x[:,1])
        state = self.to_local_state(x[:,2])
        action = torch.cat((out_local_region, out_local_point), dim =1)
        return x, action, state, feature

    def forward_decoder(self, x):
        # embed tokens
        x = x[:, :67, :]
        x = self.decoder_embed(x) # 全连接层，将encoder降为为decoder的输入

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove actions and state token
        x = x[:, 3:, :]

        return x
    
    def global_to_local(self, global_point, car_pos):
        '''
        Calculate the local position of a point in the global map
        '''
        local_x = global_point[:, 0] - car_pos[:, 0]
        local_y = global_point[:, 1] - car_pos[:, 1]

        center = torch.tensor([self.map_real_w / 2, self.map_real_h / 2])
        local_x += center[0]
        local_y += center[1]

        return torch.stack((local_x, local_y), dim=1)
    
    def local_to_global(self, local_point, car_pos):
        '''
        Calcluate the global position of the goal from local map
        '''
        map_w = 125
        map_h = 125
        center = torch.tensor([map_w / 2, map_h / 2])
        x = local_point[:, 0] - center[0]
        y = local_point[:, 1] - center[1]
        return torch.stack((car_pos[:, 0] + x, car_pos[:, 1] + y), dim=1)
    
    
    def forward_loss(self, state, out_state, out_action, value, gt_value):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # action loss
        region_dist =  torch.distributions.categorical.Categorical(logits=out_action[:,:64])
        real_point = state[:,-2:]
        real_point = self.global_to_local(real_point, state)
        real_point[:, 0] = torch.clamp(real_point[:, 0], max=self.map_real_w-1e-5, min = 0.01)  
        real_point[:, 1] = torch.clamp(real_point[:, 1], max=self.map_real_h-1e-5, min = 0.01) 
        real_regions = real_point // torch.Tensor([self.map_real_w/self.grid_size, self.map_real_h/self.grid_size]).to(real_point.device)
        gt_regions = (real_regions[:,1] + self.grid_size * real_regions[:,0]).to(torch.int64)
        region_loss = F.cross_entropy(out_action[:,:64], gt_regions)
        # Critic loss
        critic_loss = F.smooth_l1_loss(value, gt_value, beta=10)
        result = {}
        result['regions_loss'] = region_loss
        result['critic_loss'] = critic_loss
        
        # sum all loss
        loss =  region_loss + critic_loss
        return loss, result  

    
    def forward(self, imgs, state, gt_value, mask_ratio=0.75):
        
        latent, out_action, out_state, value = self.forward_encoder(imgs)
        # pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss, result = self.forward_loss(state, out_state, out_action, value, gt_value)
        return loss, result
    
class MultiAgentAutoencoder_nextstate(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone, 默认vit-large
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # parameter
        self.n_agents = 3
        self.q_flag = False
        self.grid_size = 8
        self.map_real_w = 125
        self.map_real_h = 125
        self.mean = torch.Tensor([0.0474, 0.171, 0.0007])
        self.std = torch.Tensor([0.4430, 0.3323, 0.7])
        self.max_speed = 8
        self.max_theta = math.pi/3
        self.state_scale = torch.tensor([self.map_real_w, self.map_real_h, self.max_theta])
        self.dw = self.map_real_w/self.grid_size/2
        self.dh = self.map_real_h/self.grid_size/2
        # model(encoder)
        self.embed_dim = 128
        drop_rate = 0.
        norm_layer=nn.LayerNorm
        self.patch_embed = PatchEmbed(
            img_size=128, patch_size=16, in_chans=3, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.action_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
        self.state_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, self.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-2)])
        self.ca_blocks = nn.ModuleList([
            CA_Block(self.embed_dim, num_heads = 16, mlp_ratio = 4., qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.norm = norm_layer(self.embed_dim)
        self.to_local_region = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.to_local_point  = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )
        self.others_norm = norm_layer(self.embed_dim)
        self.to_local_state = nn.Linear(self.embed_dim, self.map_real_w+self.map_real_h)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=16, stride=16, bias=False)
        self.mask_conv.weight.data.fill_(1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_encoder(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.action_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight) # 均匀分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # 常数为0初始化
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs): # 图片划分为块
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x): # 块返回为图片
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # 需要保留块的数目
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio = None, nei = None):
        self.B = x.shape[0]
        if nei is None:
            # calculate Invalid Action Mask 
            agent_boundary_map = x[:,0,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            # encoder
            x = rearrange(x, 'B n c h w -> (B n) c h w')    
            x = self.patch_embed(x)
        else:
            agent_boundary_map = x[:,1] * self.std[1] + self.mean[1] # boundary map
            self.IAM_mask = self.mask_conv(agent_boundary_map.unsqueeze(1)).view(self.B,-1) < 100
            x = self.patch_embed(x)
            x = torch.cat((x.unsqueeze(1), nei), dim = 1)
            x = rearrange(x, 'B n p l -> (B n) p l')
        # x = rearrange(x, '(B n) p l -> B (n p) l', B = self.B)
        action_token = self.action_token + self.pos_embed[:, :2, :]
        action_token = action_token.expand(x.shape[0], -1, -1)
        state_token = self.state_token + self.pos_embed[:, 2, :]
        state_token = state_token.expand(x.shape[0], -1, -1)
        x = torch.cat((action_token, state_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        # cross attention
        x = rearrange(x, '(B n) p l -> B n p l', B = self.B)    
        agent = x[:, 0]
        others = self.others_norm(rearrange(x[:, 1:], 'B n p l -> B (n p) l'))
        for blk in self.ca_blocks:
            agent = blk(agent, others)
        x = agent
        x = self.norm(x) # x.shape = [B d l] = [32 1178 128]
        out_local_region = self.to_local_region(x[:,0])
        out_local_region = torch.where(self.IAM_mask, out_local_region, torch.tensor(-1e4).to(dtype=out_local_region.dtype, device=out_local_region.device))
        out_local_point = self.to_local_point(x[:,1])
        state = self.to_local_state(x[:,2])
        action = torch.cat((out_local_region, out_local_point), dim =1)
        return x, action, state

    def forward_decoder(self, x):
        # embed tokens
        x = x[:, :67, :]
        x = self.decoder_embed(x) # 全连接层，将encoder降为为decoder的输入

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove actions and state token
        x = x[:, 3:, :]

        return x
    
    def global_to_local(self, global_point, car_pos):
        '''
        Calculate the local position of a point in the global map
        '''
        local_x = global_point[:, 0] - car_pos[:, 0]
        local_y = global_point[:, 1] - car_pos[:, 1]

        center = torch.tensor([self.map_real_w / 2, self.map_real_h / 2])
        local_x += center[0]
        local_y += center[1]

        return torch.stack((local_x, local_y), dim=1)
    
    def local_to_global(self, local_point, car_pos):
        '''
        Calcluate the global position of the goal from local map
        '''
        map_w = 125
        map_h = 125
        center = torch.tensor([map_w / 2, map_h / 2])
        x = local_point[:, 0] - center[0]
        y = local_point[:, 1] - center[1]
        return torch.stack((car_pos[:, 0] + x, car_pos[:, 1] + y), dim=1)
    
    
    def forward_loss(self, gt, state, out_state):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # state loss
        agent_next_state = gt.to(state.device, torch.int64)
        H_loss = F.cross_entropy(out_state[:,:self.map_real_w], agent_next_state[:,0])
        W_loss = F.cross_entropy(out_state[:,self.map_real_w:], agent_next_state[:,1])
        
        loss = W_loss + H_loss
        result = {}
        result['state_loss'] = loss
        
        return loss, result  

    
    def forward(self, imgs, state, gt, mask_ratio=0.75):
        
        latent, out_action, out_state = self.forward_encoder(imgs)
        # pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss, result = self.forward_loss(gt, state, out_state)
        return loss, result
    

# ----------------------------------------------------------------------------------------------
def maae_base(**kwargs):
    model = MultiAgentAutoencoder(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def Baseline_1(**kwargs):
    model = MultiAgentAutoencoder_withCritic(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def Baseline_2(**kwargs):
    model = MultiAgentAutoencoder_nextstate(
        patch_size=16, embed_dim=768, depth=6, num_heads=12,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = maae_base  # decoder: 512 dim, 8 blocks
