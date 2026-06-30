# -*- coding: utf-8 -*-
# Author: Zonglin Meng 


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

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
class Single_Agent_Fuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input_dict) -> torch.Tensor:
        '''
        input_dict = { 'síngle_agent_bev_img': x,
               	'depth_items': depth_items,
               	'spatial_features': spatial_features}
        '''
        #print('inputs',inputs[0].shape, inputs[1].shape)
        #print('síngle_agent_bev_img', input_dict['síngle_agent_bev_img'].shape)
        #print('spatial_features', input_dict['spatial_features'].shape)
        try:
            multi_modality_feature = torch.cat((input_dict['síngle_agent_bev_img'], input_dict['spatial_features']), dim=1)
            #print('síngle_agent_bev_img',input_dict['síngle_agent_bev_img'].shape) #[6, 128, 200, 704]
            #print('spatial_features',input_dict['spatial_features'].shape) #[6, 64, 200, 704]
        except:
            print('síngle_agent_bev_img',input_dict['síngle_agent_bev_img'].shape)
            print('spatial_features',input_dict['spatial_features'].shape)
        #print('multi_modality_feature',multi_modality_feature.shape)
        return super().forward(multi_modality_feature)

class MMIntermediate(LiftSplatShoot):
    def __init__(self, args): 
        super(MMIntermediate, self).__init__(args)

        #print('##########################################')
        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.bevencode = BevEncodeSSFusion(fusion_args)
        
        lidar_args = args['lidar_args']
        self.pillar_vfe = PillarVFE(lidar_args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])

        self.scatter = PointPillarScatter(lidar_args['point_pillar_scatter'])
        
        #self.supervise_single = args['supervise_single']
        self.single_agent_fusion = Single_Agent_Fuser(lidar_args['point_pillar_scatter']['num_features'] + args['bevout_feature'], args['single_agent_fusion']['out_channels'])
        for p in self.camencode.parameters():
            p.requires_grad_(False)
        for p in self.camencode_range_view.parameters():
            p.requires_grad_(False)
        #if self.supervise_single:
        #    self.cls_head_before_fusion = nn.Conv2d(self.bevout_feature, args['anchor_number'], kernel_size=1)                 
        #    self.reg_head_before_fusion = nn.Conv2d(self.bevout_feature, 7 * args['anchor_number'], kernel_size=1)
        #    if self.use_dir:
        #        self.dir_head_before_fusion = nn.Conv2d(self.bevout_feature, args['dir_args']['num_bins'] * args['anchor_number'], kernel_size=1) 
    
    def forward(self, data_dict):
        
        single_agent_dict = self.extract_single_agent_feature(data_dict)
        multi_modality_feature = self.single_agent_fusion(single_agent_dict)
        
        multi_agent_dict = {
        'multi_modality_feature':multi_modality_feature,
        'pairwise_t_matrix': data_dict['pairwise_t_matrix'],
        'record_len': data_dict['record_len'],
        'depth_items': single_agent_dict['depth_items']
        }

        data_dict = self.multi_agent_feature_fusion(multi_agent_dict)

        return data_dict
    
    
    #def multi_agent_feature_fusion(self, input_dict, pairwise_t_matrix, record_len):
    def multi_agent_feature_fusion(self, input_dict):
        fuse_feature = input_dict['multi_modality_feature']
        #img_feature = input_dict['síngle_agent_bev_img']
        depth_items = input_dict['depth_items']
        pairwise_t_matrix = input_dict['pairwise_t_matrix']
        record_len = input_dict['record_len']
        x_single, x_fuse = self.bevencode(fuse_feature, record_len, pairwise_t_matrix)
        #print('x2', x_fuse.shape) #[1, 128, 120, 120]
        #print('x3', x_single.shape) #[2, 128, 120, 120]
        psm = self.cls_head(x_fuse)
        #print('psm',psm.shape)
        rm = self.reg_head(x_fuse)
        #print('rm',rm.shape)
        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'depth_items': depth_items}
        return output_dict
    
    def extract_single_agent_feature(self, input_dict):
        '''
        input: Images for each agent.
               Batch, 
        
        '''
        
        x, depth_items = self.extract_img_feature(input_dict)
        #output_dict = {'síngle_agent_bev_img': x,
        #               'depth_items':depth_items}
        
        spatial_features = self.extract_lidar_feature(input_dict)
        #print('x',x.shape) #[2, 128, 240, 240]
        #print('spatial_features',spatial_features.shape) #[2, 64, 240, 240]
        output_dict = {'síngle_agent_bev_img': x,
                       'depth_items': depth_items,
                       'spatial_features': spatial_features}
        return output_dict

        
    def extract_img_feature(self, input_dict):
        #print('#######################################')
        image_inputs_dict = input_dict['image_inputs']
        #record_len = input_dict['record_len']

        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        
        lidar2img = image_inputs_dict['lidar2img']
        lidar = input_dict['lidar_np']
        x_range_view = self.get_depth_feature(x, lidar, lidar2img)
        # lss get voxels and depth
        #print('x',x.shape) #[5, 4, 3, 480, 640] N_vehicle, N_views, C, W, H
        x, depth_items = self.get_voxels(x, x_range_view, rots, trans, intrins, post_rots, post_trans)
        #print('x1', x.shape) #[5, 128, 240, 240] N_vehicle, C, W, H

        return x, depth_items
    
    def extract_lidar_feature(self, input_dict):
        voxel_features = input_dict['processed_lidar']['voxel_features']
        #print('voxel_features',voxel_features.shape) #[17414, 32, 4]
        voxel_coords = input_dict['processed_lidar']['voxel_coords']
        voxel_num_points = input_dict['processed_lidar']['voxel_num_points']
        record_len = input_dict['record_len']
        lidar_pose = input_dict['lidar_pose']
        pairwise_t_matrix = input_dict['pairwise_t_matrix']
        
        lidar_input_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}
            


        lidar_input_dict = self.pillar_vfe(lidar_input_dict)
        lidar_input_dict = self.scatter(lidar_input_dict)
        return lidar_input_dict['spatial_features']
def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShootIntermediate(grid_conf, data_aug_conf, outC)
