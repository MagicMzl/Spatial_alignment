# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from numpy import record
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from icecream import ic
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, BevEncodeSSFusion, Up, CamEncode, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt


class LiftSplatShootIntermediate(LiftSplatShoot):
    def __init__(self, args): 
        super(LiftSplatShootIntermediate, self).__init__(args)

        #print('##########################################')
        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.bevencode = BevEncodeSSFusion(fusion_args)
        self.supervise_single = args['supervise_single']

        for p in self.camencode.parameters():
            p.requires_grad_(False)
        for p in self.camencode_range_view.parameters():
            p.requires_grad_(False)
        
        
        if self.supervise_single:
            self.cls_head_before_fusion = nn.Conv2d(self.bevout_feature, args['anchor_number'], kernel_size=1)                 
            self.reg_head_before_fusion = nn.Conv2d(self.bevout_feature, 7 * args['anchor_number'], kernel_size=1)
            if self.use_dir:
                self.dir_head_before_fusion = nn.Conv2d(self.bevout_feature, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) 
    
    def forward(self, data_dict):
        return self._forward(data_dict)

    def _forward(self, data_dict):
        #print('#######################################')
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        
        x, rots, trans, intrins, post_rots, post_trans, lidar2img = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans'], image_inputs_dict['lidar2img']
            
        lidar = data_dict['lidar_np']
        #print('lidar2img',lidar2img.shape)
        x_range_view = self.get_depth_feature(x, lidar, lidar2img)
        
        #x_depth = torch.cat((x, x_range_view), dim = 2)
        
        # lss get voxels and depth
        #print('x',x.shape) #[5, 4, 3, 480, 640] N_vehicle in batch, N_views, C, W, H
        #print('record_len',record_len) #[3,2]
        #print('x_range_view 1',x_range_view.shape)
        
        x, depth_items = self.get_voxels(x, x_range_view, rots, trans, intrins, post_rots, post_trans)
        #print('x1', x.shape) #[5, 128, 240, 240] N_vehicle, C, W, H
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_single, x_fuse = self.bevencode(x, record_len, pairwise_t_matrix)
        #print('x2', x_fuse.shape) #[1, 128, 120, 120]
        #print('x3', x_single.shape) #[2, 128, 120, 120]
        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'depth_items': depth_items}


        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShootIntermediate(grid_conf, data_aug_conf, outC)
