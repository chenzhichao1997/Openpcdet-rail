import torch
from torch import nn
from ....utils.spconv_utils import replace_feature, spconv
import numpy as np
from functools import partial
class Mobilelike(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, norm_fn=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01), downsample=None, indice_key=None,expansion = 3,ksize=3):
        super(Mobilelike, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        self.ksize = ksize
        if self.ksize==3:
            self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key,
        )
        else:
            self.conv1 = nn.Linear(inplanes,planes)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()


    def forward(self, lidar_feature,img_feature):
        x = torch.cat([lidar_feature.features,img_feature],dim=-1)
        x = replace_feature(lidar_feature,x)
        if self.ksize==3:
            out = self.conv1(x)#根据pointnet的思路，共享的fc层会让点与点之间没有信息交互，所以共享fc后接入到稀疏卷积，让特征在空间维度上能够交互.
            #点云特征提取器采用多层感知机（MLP）对每个点进行特征提取。MLP可以对每个点的坐标和其他信息进行非线性变换，从而提取出更具代表性的特征。
            # 通过这种方式，PointNet网络能够捕捉到点云数据的局部细节和几何结构。
        else:
            out = replace_feature(x,self.conv1(x.features))
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out


class CMAFM(nn.Module):
    def __init__(self,lidar_channel,cam_channel,out_channel=256):
        super(CMAFM,self).__init__()
        channels_all = 2*(lidar_channel+cam_channel)
        self.mlp_lidar = nn.Sequential(nn.Linear(channels_all,channels_all//3,bias=False),
                                       nn.ReLU(),
                                       nn.Linear(channels_all // 3,lidar_channel, bias=False),
                                       nn.ReLU(),
                                  )
        self.mlp_cam = nn.Sequential(nn.Linear(channels_all,channels_all//3,bias=False),
                                       nn.ReLU(),
                                       nn.Linear(channels_all // 3,cam_channel, bias=False),
                                       nn.ReLU(),
                                  )
        self.mlp_fusion = nn.Sequential(nn.Linear(channels_all//2, out_channel, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(out_channel, out_channel, bias=False),
                                    nn.ReLU(),
                                     )
    def split_tensor(self,x,indices_cat):
        batch_index = indices_cat[:, 0]
        unique_elements, counts = torch.unique(batch_index, return_counts=True)
        split_tensors = torch.split(x, tuple(counts.tolist()))
        return split_tensors

    def forward(self,lidar_feat,cam_feat):
        lidar_list = self.split_tensor(lidar_feat.features, lidar_feat.indices)
        cam_list = self.split_tensor(cam_feat, lidar_feat.indices)
        out = []
        for lidar_feature,cam_feature in zip(lidar_list,cam_list):
            lidar_mean = lidar_feature.min(dim=0, keepdim=True)[0]
            lidar_max = lidar_feature.max(dim=0, keepdim=True)[0]
            cam_mean = cam_feature.min(dim=0, keepdim=True)[0]
            cam_max = cam_feature.max(dim=0, keepdim=True)[0]
            cat_mean_max = torch.cat([lidar_mean,lidar_max,cam_mean,cam_max],dim=-1)
            lidar_attentation = torch.sigmoid(self.mlp_lidar(cat_mean_max))
            cam_attentation = torch.sigmoid(self.mlp_cam(cat_mean_max))
            lidar_feature = lidar_feature*lidar_attentation
            cam_feature = cam_feature*cam_attentation
            lidar_feature = torch.cat([lidar_feature,cam_feature],dim=-1)
            lidar_feature = self.mlp_fusion(lidar_feature)
            out.append(lidar_feature)
        out = torch.cat(out,dim=0)
        lidar_feat = lidar_feat.replace_feature(out)
        return lidar_feat






class Sparse_ConvFuser(nn.Module):
    def __init__(self,model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        lidar_inchannel = self.model_cfg.IN_CHANNEL_LIDAR
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.lidar_only = self.model_cfg.lidar_only
        if self.lidar_only:
            self.conv = spconv.SubMConv2d(lidar_inchannel, out_channel, 1, bias=False, indice_key='conv')



        if self.model_cfg.get('Fusion', None) is not None:
            self.fusion = CMAFM(lidar_inchannel,in_channel-lidar_inchannel,out_channel)
        else:
            self.fusion = Mobilelike(in_channel,out_channel,ksize=1)
        self.cam_only = self.model_cfg.get('cam_only', None)
        pool_type = self.model_cfg.get('pool_type', None)
        pool_size = self.model_cfg.get('pool_size',None)
        self.pool_type = pool_type
        if pool_type is not None:
            if pool_type == 'max':
                self.sp_pool = nn.MaxPool2d(kernel_size=pool_size,stride=1,padding= pool_size//2)
            elif pool_type == 'mean':
                self.sp_pool = nn.AvgPool2d(kernel_size=pool_size,stride=1,padding=1)
            elif pool_type == 'None':
                self.sp_pool = nn.MaxPool2d(kernel_size=pool_size,stride=1,padding=0)




    # def transform_to_sparse(self,x):
    #     # batch_size,c,h,w = x.shape
    #     # pillar_features = x.reshape(batch_size,c,-1).permute(0, 2, 1).reshape(-1, c).contiguous()
    #     # sparse_shape = np.array([h,w])
    #     # #create index
    #     # batch_ids = torch.arange(batch_size).repeat_interleave(h*w).to(pillar_features.device)
    #     # x_coords = torch.arange(w).repeat(h, batch_size).view(-1).to(pillar_features.device)
    #     # y_coords = torch.arange(h).repeat(w,batch_size).t().contiguous().view(-1).to(pillar_features.device)
    #     # coords = torch.stack([batch_ids,y_coords,x_coords],dim=1).int()
    #     # x = spconv.SparseConvTensor(
    #     #     features=pillar_features,
    #     #     indices=coords,
    #     #     spatial_shape=sparse_shape,
    #     #     batch_size=batch_size
    #     # )
    #     x = x.permute(0,2,3,1).contiguous()
    #     print(x.size())
    #     x = spconv.SparseConvTensor.from_dense(x)
    #     return x



    def sparse_fusion(self,lidar_feature,img_feature):
        img_feature = img_feature.permute(0,2,3,1).contiguous()
        shape = img_feature.shape
        img_feature = img_feature.view(shape[0]*shape[1]*shape[2],shape[3])
        indices = lidar_feature.indices#获取lidar的indices [n',3]
        spatial_shape = lidar_feature.spatial_shape
        spatial_size = lidar_feature.spatial_size

        # # d = [batchid +1] * [x + y*w]
        batch_indices, x_coords, y_coords  = indices[...,0],indices[...,2],indices[...,1]
        row_indices = (batch_indices*spatial_size)+(y_coords*spatial_shape[1]+x_coords)
        row_indices = torch.clamp(row_indices,min=0,max=spatial_size-2)
        row_indices = row_indices.long().tolist()

        img_feature = img_feature[row_indices,:].contiguous()#用最大池化来实现邻域特征选择.或者卷积.
        if self.cam_only is not None:
            lidar_feature = replace_feature(lidar_feature,lidar_feature.features.zero_())
        lidar_feature = self.fusion(lidar_feature,img_feature)
        return lidar_feature
    def forward(self,batch_dict):
        if self.lidar_only:
            batch_dict['encoded_spconv_tensor']=self.conv(batch_dict['encoded_spconv_tensor'])
            return batch_dict
        """
        Args:
            batch_dict:
                spatial_features_img (tensor): Bev features from image modality
                spatial_features (tensor): Bev features from lidar modality

        Returns:
            batch_dict:
                spatial_features (tensor): Bev features after muli-modal fusion
        """
        img_bev = batch_dict['spatial_features_img']
        if self.pool_type:
            img_bev = self.sp_pool(img_bev)

        lidar_bev = batch_dict['encoded_spconv_tensor']



        fusion_feature = self.sparse_fusion(lidar_bev,img_bev)
        batch_dict['encoded_spconv_tensor'] = fusion_feature #Spconv_tensor n',256
        return batch_dict