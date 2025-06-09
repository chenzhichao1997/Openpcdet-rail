from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv
import matplotlib.pyplot as plt
import numpy as np

class SP_Attentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SP_Attentation, self).__init__()
        self.fc_query = nn.Linear(in_channels, out_channels)
        self.fc_key = nn.Linear(in_channels, out_channels)
        self.fc_value = nn.Linear(in_channels, out_channels)

    def split_tensor(self,x,indices_cat):
        batch_index = indices_cat[:, 0]
        unique_elements, counts = torch.unique(batch_index, return_counts=True)
        split_tensors = torch.split(x, tuple(counts.tolist()))
        return split_tensors

    def forward(self, x):
        tensor_list = self.split_tensor(x.features,x.indices)
        out = []
        for sigle_feature in tensor_list:
            # 获取查询、键、值
            query = self.fc_query(sigle_feature)
            key = self.fc_key(sigle_feature)
            value = self.fc_value(sigle_feature)
            # 计算注意力权重
            attn_weights = torch.matmul(query, key.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            # 应用注意力权重
            out = torch.matmul(attn_weights, value)
        out = torch.cat(out, dim=0)
        # 用替换特征函数返回结果
        return x.replace_feature(out)



def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m






class SparseBasicBlock(spconv.modules.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)
        if self.inchanel == self.outchanels:
            out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out





class Mobilelike(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None,expansion = 3,ksize=3):
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


    def forward(self, x):
        if self.ksize==3:
            out = self.conv1(x)#根据pointnet的思路，共享的fc层会让点与点之间没有信息交互，所以共享fc后接入到稀疏卷积，让特征在空间维度上能够交互.
            #点云特征提取器采用多层感知机（MLP）对每个点进行特征提取。MLP可以对每个点的坐标和其他信息进行非线性变换，从而提取出更具代表性的特征。
            # 通过这种方式，PointNet网络能够捕捉到点云数据的局部细节和几何结构。
        else:
            out = replace_feature(x,self.conv1(x.features))
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out





class Bottleneck_S1(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S1, self).__init__()
        assert norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3


        self.point_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.sparse_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion)

    def forward(self, x):
        identify = x
        x = self.point_conv(x)
        x = self.sparse_conv(x)
        x = x.replace_feature(identify.features + x.features)
        return x


class Sparse_Bev_Encoder(nn.Module):
    def __init__(self,model_cfg,input_channels):
        super(Sparse_Bev_Encoder, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        spconv_kernel_sizes = [3,3]
        self.model_cfg = model_cfg
        chanles = self.model_cfg.get('in_chanles')
        self.use_attentation = model_cfg.get('Attentation', None)
        if self.use_attentation is not None:
            self.sp_attentation1 = SP_Attentation(chanles,chanles)
            self.sp_attentation2 = SP_Attentation(chanles,chanles)
        self.conv1 = spconv.SparseSequential(
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev1'),
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(chanles, chanles, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev2'),
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(chanles, chanles, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev3'),
            Bottleneck_S1(chanles, chanles, norm_fn=norm_fn, indice_key='bev3'),
        )

        self.num_bev_features = 256


    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices
        ##############新增代码############
        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)#挑选出独立且不重复的元素,inv是每个元素在唯一值张量中的索引
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)#独立不重复进行特征的相加.

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv.spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        feat_8 = self.conv1(batch_dict['encoded_spconv_tensor'])
        if self.use_attentation is not None:
            feat_8 = self.sp_attentation1(feat_8)
        feat_16 = self.conv2(feat_8)
        if self.use_attentation is not None:
            feat_16 = self.sp_attentation2(feat_16)
        feat_32 = self.conv3(feat_16)
        feat_16.indices[:, 1:] *= 2
        feat_32.indices[:, 1:] *= 4
        feat_8 = feat_8.replace_feature(torch.cat([feat_8.features, feat_16.features, feat_32.features]))
        feat_8.indices = torch.cat([feat_8.indices, feat_16.indices, feat_32.indices])

        batch_dict['encoded_spconv_tensor'] = self.bev_out(feat_8)

        #self.vis_feature(batch_dict)


        return batch_dict

    def vis_feature(self,x):
        feature_map=x['encoded_spconv_tensor'].dense()
        feature_map = feature_map.squeeze(0)
        feature_map_mean = torch.mean(feature_map,dim=0)
        feature_map_mean = feature_map_mean.transpose(0,1)
        feature_map_mean = 128*feature_map_mean.cpu().numpy()
        plt.imshow(feature_map_mean, cmap='viridis')  # 使用伪彩色图
        import os
        plt.savefig(os.path.join("/home/G9120130136/userdata/czc/OpenPCDet-master/vis",x['frame_id'][0]+".jpg"))


        return 0

