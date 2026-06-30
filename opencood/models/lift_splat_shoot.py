# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization
from opencood.models.sub_modules.lss_submodule import Up, CamEncode, BevEncode, CamEncode_Resnet101, CamEncode_range_view
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

class SelfAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttentionFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Reduce the number of channels before self-attention
        self.reduce_channels = nn.Linear(in_channels, in_channels // 2)
        self.restore_channels = nn.Linear(in_channels // 2, out_channels)
        
        self.query = nn.Linear(in_channels // 2, out_channels // 2)
        self.key = nn.Linear(in_channels // 2, out_channels // 2)
        self.value = nn.Linear(in_channels // 2, out_channels // 2)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), 1)
        batch_size, channels, *spatial_dims = x_cat.size()
        spatial_size = torch.prod(torch.tensor(spatial_dims)).item()
        
        x_flat = x_cat.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Reduce channels
        x_flat = self.reduce_channels(x_flat)
        
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        attention = self.softmax(scores)
        
        out = torch.matmul(attention, V)
        
        out = self.restore_channels(out)
        out = out.permute(0, 2, 1).view(batch_size, self.out_channels, *spatial_dims)
        return out
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)
        attention_weights = self.attention(x_cat)
        attention_weights = attention_weights.unsqueeze(2)  # Match the channel dimension
        
        x_fuse = attention_weights[:, 0:1] * x1 + attention_weights[:, 1:2] * x2
        return x_fuse
        
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension
        gate = self.gate(x_cat)  # Generate gate values between 0 and 1
        x_fuse = gate * x1 + (1 - gate) * x2
        return x_fuse



class LiftSplatShoot(nn.Module):
    def __init__(self, args): 
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        self.bevout_feature = args['bevout_feature']
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )  # 划分网格

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda")) 
        
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)

        self.D, _, _, _ = self.frustum.shape 
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode_range_view = CamEncode_range_view(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])

        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(self.bevout_feature, args['anchor_number'],
                                  kernel_size=1)                 
        self.reg_head = nn.Conv2d(self.bevout_feature, 7 * args['anchor_number'],
                                  kernel_size=1)
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.bevout_feature, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        
        #self.self_attention_fusion = SelfAttentionFusion(in_channels=256, out_channels=256)
        #self.attention_fusion = AttentionFusion(channels=128)
        self.GatedFusion = GatedFusion(channels = 128)
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim'] 
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  

        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)
        
        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        #print('points',points.shape)
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], 
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        #print('cam coord',points.shape)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        #print('rots',rots)
        #print('tranns',trans)
        #rots_noise = rots[:, 0] + 0.5
        #print('rots_noise',rots_noise)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        return points  

    def get_cam_feats(self, x, x_range_view):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        
        x = x.view(B*N, C, imH, imW)
        
        x_range_view = x_range_view.view(B*N, -1, imH, imW)
        #print('x_range_view', x_range_view.shape) #[12, 1, 480, 640]
        #print('x', x.shape) #[12, 3, 480, 640]
        depth_items_range_view, x_range_view = self.camencode_range_view(x_range_view)
        
        depth_items, x = self.camencode(x)
        
        #print('x_range_view 1',x_range_view.shape) #[24, 128, 48, 60, 80]
        #print('x 1',x.shape) #[24, 128, 48, 60, 80]
        #print('####################################')
        #x_cat = torch.cat((x, x_range_view), 1)
        #x_fuse = self.self_attention_fusion(x, x_range_view)
        #x_fuse = self.attention_fusion(x, x_range_view)
        
        x_fuse = self.GatedFusion(x, x_range_view)
        
        #print('x_fuse',x_fuse.shape)
        #print('####################################')
        x = x_fuse.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        
        x = x.permute(0, 1, 3, 4, 5, 2)
        
        return x, depth_items

    def voxel_pooling(self, geom_feats, x):

        B, N, D, H, W, C = x.shape 
        Nprime = B*N*D*H*W 

        # flatten x
        x = x.reshape(Nprime, C)  

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3) 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) 

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  


        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  

        
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        #print('final',final.shape)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中
        
        #print('image final',final.shape) #[2, 128, 1, 200, 704]
        # collapse Z
        
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维
        #print('final2',final.shape) #[2, 128, 1, 200, 704]
        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W
        
        
    def get_depth_feature(self, img, lidar, lidar2img):
        img_size = img.shape
        batch_size = img_size[0]
        depth = torch.zeros(img_size[0], img_size[1], 1, img_size[3], img_size[4]).cuda() # 创建大小
        lidar2img = lidar2img.view(img_size[0],img_size[1],4,4)
        #rots_depth = []
        #trans_depth = []
        #print('lidar',len(lidar))
        #print('img_size',img_size)
        for b in range(batch_size):
            #print('lidar b',lidar[b].shape)
            lidar[b] = torch.from_numpy(lidar[b]).to(img.get_device())
            cur_coords = lidar[b][:, :3].float()  #取点的xyz
            #print('cur_coords 1',cur_coords.shape)
            # lidar2image
            cur_coords = lidar2img[b][:, :3, :3].matmul(cur_coords.transpose(1, 0))
            #print('lidar2img',lidar2img[b][:3, :3].shape)
            #print('cur_coords 2',cur_coords.shape)
            cur_coords += lidar2img[b][:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            #print('cur_coords 3',cur_coords.shape)
            # imgaug

            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]
            #print('cur_coords 4',cur_coords.shape)
            on_img = (
                    (cur_coords[..., 0] < img_size[3])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < img_size[4])
                    & (cur_coords[..., 1] >= 0)
            )
            #print('on_img',on_img.shape)
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()  # 点云投影到图像坐标
                masked_dist = dist[c, on_img[c]]  # 对应深度
                #print('c',c, on_img.shape[0])
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist.float()  # 稀疏的深度约束图（用于计算loss） 1, 6, 1, 448, 800
            
        step = 7
        B, N, C, H, W = depth.size()
        depth_tmp = depth.reshape(B*N, C, H, W)
        pad = int((step - 1) // 2)
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [2, 6, 1, 256, 704]
        #img_metas[0].update({'max_depth': max_depth})

        # 求解max_depth四个方向梯度, 随后concat depth, 以缓解深度跳变对深度预测模块的影响
        step = float(step)
        shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
        max_depth_tmp = max_depth.reshape(B*N, C, H, W)
        output_list = []
        for shift in shift_list:
            transform_matrix =torch.tensor([[1, 0, shift[0]],[0, 1, shift[1]]]).unsqueeze(0).repeat(B*N, 1, 1).cuda()
            grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
            output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)  #平移后图像
            output = max_depth - output
            output_mask = ((output == max_depth) == False)
            output = output * output_mask
            output_list.append(output)
        grad = torch.cat(output_list, dim=2)  # [2, 6, 4, 256, 704]
        max_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)
        #img_metas[0].update({'max_grad': max_grad})
        #depth_ = depth
        depth = torch.cat([depth, grad], dim=2)  # [1, 6, 5, 448, 800]
        #print('depth',depth.shape)
            
        return depth
    def get_voxels(self, x, x_range_view, rots, trans, intrins, post_rots, post_trans):
        #print('x',x.shape) #[2, 4, 3, 480, 640]
        #print('rot',rots.shape) #[2, 4, 3, 3] 
        #  [ 6.4590e-08, -4.6890e-10,  1.0000e+00],
        #  [ 1.0000e+00, -1.9890e-10, -6.4613e-08],
        #  [-1.7296e-10, -1.0000e+00, -2.7855e-10]
        #print('trans',trans.shape) #[2, 4, 3]
        # [ 3.0000e+00, -2.0473e-06, -9.0000e-01]
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) 
        #print('x_depth 2',x.shape)
        x_img, depth_items = self.get_cam_feats(x, x_range_view) 
        x = self.voxel_pooling(geom, x_img) 

        return x, depth_items

    def forward(self, data_dict):
        image_inputs_dict = data_dict['image_inputs']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        #print('get voxel',x.shape)
        x = self.bevencode(x)
        if self.shrink_flag:
            x = self.shrink_conv(x)
        psm = self.cls_head(x)
        rm = self.reg_head(x)
        output_dict = {'cls_preds': psm,
                       'reg_preds': rm,
                       'depth_items': depth_items}
        if self.use_dir:
            dm = self.dir_head(x)
            output_dict.update({"dir_preds": dm})
        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
