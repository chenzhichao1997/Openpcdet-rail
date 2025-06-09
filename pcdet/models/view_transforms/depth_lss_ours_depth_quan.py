import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool
import numpy as np
import cv2

def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(scalar_tensor, torch.Tensor):
        scalar_tensor = torch.from_numpy(scalar_tensor)
     # 确保张量是连续的
    scalar_tensor = scalar_tensor.contiguous()
    to_use = scalar_tensor.view(-1)
    while to_use.shape[0] > 2 ** 24:
        to_use = to_use[::2]
    with torch.inference_mode(False):
        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

    scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    scalar_tensor = scalar_tensor.clamp_(0, 1)

    scalar_tensor = ((1 - scalar_tensor) * 255).byte().cpu().numpy()  # inverse heatmap
    return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class DepthLSSTransform_ours_depth_quan(nn.Module):
    """
        This module implements LSS, which lists images into 3D and then splats onto bev features.
        This code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE
        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND
        downsample = self.model_cfg.DOWNSAMPLE

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        # dx每个维度上的网格间距;(0.3,0.3,20)
        # bx每个维度上的起点的中心点(0.15,-53.85,0)
        # nx每个维度上的网格数量(360,360,1)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        # 在原图上的像素上创建了一系列的射线. 根据特征图的大小进行步长选择.
        # 例如特征图大小是30,40. 相当于在图像上每30pix上就有一条距离的射线.
        self.D = self.frustum.shape[0]

        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # nn.Conv2d(8, 32, 1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.Conv2d(32, 64, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channel + 64 , in_channel, 3, padding=1),## + 64
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # nn.Conv2d(in_channel, self.D + self.C, 1),### self.D + self.C
            # nn.BatchNorm2d(self.D + self.C),
            # nn.ReLU(True),
                        
            nn.Conv2d(in_channel, self.D , 1),### self.D + self.C
            nn.BatchNorm2d(self.D ),
            nn.ReLU(True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        self.features_out = nn.Sequential(
            nn.Conv2d(in_channel + 64 , in_channel, 3, padding=1),## + 64
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),

            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(in_channel,  self.C, 1),### self.D + self.C
            nn.BatchNorm2d( self.C),
            nn.ReLU(True),
            
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.upsample_depth = nn.Sequential(
            nn.ConvTranspose2d(self.D , self.D, kernel_size=5, padding=2, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.D),
            nn.ReLU(),
            nn.ConvTranspose2d(self.D, self.D, kernel_size=5, padding=2, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.D),
            nn.ReLU(),
            nn.ConvTranspose2d(self.D, self.D, kernel_size=5, padding=2, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(self.D),
            nn.ReLU(),
            nn.Conv2d(self.D, self.D, kernel_size=1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()
    def create_frustum(self):
        iH, iW = self.image_size #240,320
        fH, fW = self.feature_size #30,40
        # stride = iH/fH = 8
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        #按递增的情况来生成118个距离,然后按照特征图大小扩充. size = (118,30,40)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) #对像素进行采样
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) #对像素进行采样.
        frustum = torch.stack((xs, ys, ds), -1)
        # print('frustum.shape',frustum.shape)

        # print('frustum',frustum)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):

        camera2lidar_rots = camera2lidar_rots.to(torch.float)
        camera2lidar_trans = camera2lidar_trans.to(torch.float)
        intrins = intrins.to(torch.float)
        post_rots = post_rots.to(torch.float)
        post_trans = post_trans.to(torch.float)

        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1) \
                .matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float)#4,1,118,32,88,3
        x = x.to(torch.float)
        # print('lxx',x.shape)
        B, N, D, H, W, C = x.shape #4,1,118,30,40,80
        Nprime = B * N * D * H * W
        # print(Nprime)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        # print(geom_feats.shape)

        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )



        x = x[kept]
        # print('x[kept]',x.shape)

        geom_feats = geom_feats[kept]
        # print('xxxxxxxxx',x.shape)
        # print('geom_feats,geom_feats,geom_feats',geom_feats.shape)

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    # def get_cam_feats(self, x, d):
    #     B, N, C, fH, fW = x.shape

    #     d = d.view(B * N, *d.shape[2:])#4,64,32,88
    #     x = x.view(B * N, C, fH, fW)#4,256,30,40
        
        


    #     d = self.dtransform(d)
    #     x = torch.cat([d, x], dim=1)
    #     x = self.depthnet(x)
        
    #     depth_8u = self.upsample_depth(x)
    #     depth_8u = depth_8u[:, : self.D].softmax(dim=1)
    #     depth_pre = depth_8u.contiguous()
        
    #     depth = x[:, : self.D].softmax(dim=1)

    #     x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

    #     x = x.view(B, N, self.C, self.D, fH, fW)
    #     x = x.permute(0, 1, 3, 4, 5, 2)
    #     # print('x_de',x.shape)
    #     return x, depth_pre
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])#4,64,32,88
        x = x.view(B * N, C, fH, fW)#4,256,30,40
        
        


        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        depth = self.depthnet(x)


        x = self.features_out(x)
        
        depth_8u = self.upsample_depth(depth)
        depth_8u = depth_8u.softmax(dim=1)
        depth_pre = depth_8u.contiguous()
        
        depth = depth.softmax(dim=1)

        # x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)
        x = depth.unsqueeze(1) * x.unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        # print('x_de',x.shape)
        # print('depth_pre',depth_pre.shape)

        return x, depth_pre

    def depth_lable(self, d):
        
        # torch.arange(*self.dbound, dtype=torch.float)
        # 创建 depth_bins
        # 每隔 8 个像素进行采样，得到新的张量大小为 [2, 1, 30, 40]
        # d = d[:, :, :, ::8, ::8]
        # depthimg=  d[0, 0, 0, :, :]
        # print("Min:", d.min().item())
        # print("Max:", d.max().item())
        # print("Mean:", d.mean().item())
        # print(depthimg.shape)
        # scalarimg = visualize_scalars(depthimg)    # 进行scalar调节
        # cv2.imwrite(f'./depth_l.jpg', scalarimg)    # cv2写出深度图
        depth_bins = torch.arange(*self.dbound,dtype=torch.float, device=d.device)
        # print(depth_la.squeeze(1).shape)
        # 计算每个深度值的 bin 索引
        depth_bins_indices = torch.bucketize(d.squeeze(1), depth_bins) - 1

        # 处理边界条件
        depth_bins_indices = torch.clamp(depth_bins_indices, min=0, max=len(depth_bins) - 1)
        # print(depth_bins_indices.shape)
        # 扩展 bin 数量维度
        num_bins = len(depth_bins)
        one_hot = torch.zeros(d.size(0), num_bins, *d.size()[2:], device=d.device)

        # 生成 one-hot 编码
        one_hot.scatter_(1, depth_bins_indices.unsqueeze(2), 1.0)
        one_hot = one_hot.squeeze(2).contiguous()


        return one_hot



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        x = batch_dict['image_fpn'] 
        x = x[0]
        
        B, C, H, W = x.size()
        img = x.view(B, 1, C, H, W)

        camera_intrinsics = batch_dict['camera_intrinsics']  # [4,1,4,4] 6个相机的齐次型内参矩阵.
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        points = batch_dict['points']

        batch_size = B
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            batch_mask = points[:,0] == b
            cur_coords = points[batch_mask][:, 1:4]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            # print(cur_img_aug_matrix)
            # print(cur_lidar2image)
            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # do image aug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            # filter points outside of images
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )#cam2lidar,ins,post是后处理. extra_rots是额外的lidar增强.
        # depthimg=  depth[0, 0, 0, :, :]
        # print(depthimg.shape)
        # scalarimg = visualize_scalars(depthimg)    # 进行scalar调节
        # cv2.imwrite(f'./depth.jpg', scalarimg)    # cv2写出深度图
        
        # print('depth_la',depth.shape)
        # depth_values = depth.squeeze(1).cpu().numpy()
        # print(depth_values.shape)
        # # 打印前几个深度值作为示例
        # print("深度值示例 (前两个图像的前两个位置):")

        # print("Min:", depth.min().item())
        # print("Max:", depth.max().item())
        # print("Mean:", depth.mean().item())
        # print(depth_values)  # 查看第一个 batch 的前两个位置
        # # print(depth_values[1, :2, :2])  # 查看第二个 batch 的前两个位置
        # print('depth',depth.shape)
        d_lable = self.depth_lable(depth)
        # print('depth_la',d_lable.shape)

        x, d_pre = self.get_cam_feats(img, depth)
        # print('xx',x.shape)
        # print('geom',geom.shape)

        x = self.bev_pool(geom, x)
        x = self.downsample(x)
        # convert bev features from (b, c, x, y) to (b, c, y, x)
        x = x.permute(0, 1, 3, 2)
        batch_dict['spatial_features_img'] = x
        # print('x',x.shape)
        # print('d_lable',d_lable.shape)
        # print('d_pre',d_pre.shape)
        
        batch_dict.update({
                    'd_lable': d_lable,
                })
        batch_dict.update({
                    'd_pre': d_pre,
                })

        return batch_dict