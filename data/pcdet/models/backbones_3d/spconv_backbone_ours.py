from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv

class Point_Attentation(nn.Module):

    def __init__(self, ):
        super(Point_Attentation, self).__init__()

        self.lamda = 1e-5
        self.sigmoid = nn.Sigmoid()



    def point_attentation(self,x):
        tensor_list = self.split_tensor(x.features,x.indices)
        out = []
        for sigle_feature in tensor_list:
            n,c = sigle_feature.shape
            mean = torch.mean(sigle_feature,dim=[-1],keepdim=True)
            var = torch.sum(torch.pow((sigle_feature-mean),2),dim=[-1],keepdim=True)/(n-1)
            e_t = torch.pow((sigle_feature - mean), 2) / (4 * (var + self.lamda)) + 0.5
            sigle_feature = sigle_feature + self.sigmoid(e_t) * sigle_feature
            out.append(sigle_feature)
        out = torch.cat(out,dim=0)
        x = x.replace_feature(out)
        return x



    def split_tensor(self,x,indices_cat):
        batch_index = indices_cat[:, 0]
        unique_elements, counts = torch.unique(batch_index, return_counts=True)
        split_tensors = torch.split(x, tuple(counts.tolist()))
        return split_tensors






    def forward(self, x):

        return self.point_attentation(x)



#这部分的输入通道等于输出通道,s=1
class fusion(spconv.SparseModule):
    def __init__(self, inplanes, planes,  norm_fn=None, normal_dim = 256, dropout_rate = 0.3):
        super(fusion, self).__init__()
        self.inchanel = inplanes
        self.outchanels = planes
        self.q = nn.Sequential(nn.Linear(inplanes,planes,bias=False),
                               nn.BatchNorm1d(planes) )


        self.k = nn.Sequential(nn.Linear(inplanes,planes,bias=False),
                               nn.BatchNorm1d(planes) )
        self.v = nn.Sequential(nn.Linear(inplanes,planes,bias=False),
                               nn.BatchNorm1d(planes) )
        self.mlp = nn.Sequential(
            nn.Linear(planes,inplanes,bias=False),
            nn.BatchNorm1d(inplanes),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(inplanes,inplanes,bias=False),
            nn.BatchNorm1d(inplanes),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        self.linear_fusion = nn.Sequential(nn.Linear(2*inplanes,inplanes),
                                           nn.BatchNorm1d(inplanes),
                                           nn.ReLU(inplanes)
                                           )


        # self.normal = nn.LayerNorm(inplanes)
        self.normal_dim = normal_dim

    def split_tensor(self,x,indices_cat):
        batch_index = indices_cat[:, 0]
        unique_elements, counts = torch.unique(batch_index, return_counts=True)
        split_tensors = torch.split(x, tuple(counts.tolist()))
        return split_tensors

    def two_feature_fusion(self,f_low,f_deep):
        features_low = f_low.features
        index_low = f_low.indices
        low_tensor = self.split_tensor(features_low,index_low)

        features_high = f_deep.features
        index_high = f_deep.indices
        high_tensor = self.split_tensor(features_high,index_high)
        out = []
        for low,high in zip(low_tensor,high_tensor):
            q = self.q(low)
            k = self.k(high)
            v = self.v(high)
            attn_scores = torch.matmul(q,k.T)/(self.normal_dim**0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            # attn_scores = torch.matmul(q,k.T)
            #attn_probs = torch.sigmoid(attn_scores)
            attn_output = torch.matmul(attn_probs, v)
            # out_put = self.normal(low + self.mlp(attn_output))
            out_put = self.linear_fusion(torch.cat([low,self.mlp(attn_output)],dim=-1))
            out.append(out_put)
        out = torch.cat(out,dim=0)
        f_low = f_low.replace_feature(out)
        return f_low

    def forward(self, low,high):
        x = self.two_feature_fusion(low,high)

        return x














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






class SparseBasicBlock(spconv.SparseModule):
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

class Mobilelike2(spconv.SparseModule):
    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None,expansion = 3,ksize=3):
        super(Mobilelike2, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        self.ksize = ksize
        if self.ksize=="test":
            self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=1,padding=1,bias=bias)
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



#
#
#




class Bottleneck_S3(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S3, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3
        down_scale = 2
        low_channel = planes//down_scale


        self.point_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.sparse_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion)







    def forward(self, x):
        identify = x
        x = self.point_conv(x)
        x = self.sparse_conv(x)
        x = x.replace_feature(identify.features + x.features)
        #x = x.replace_feature(self.channel_shuffle(torch.cat([csp1.features,x.features],dim=-1),2))
        return x










class Bottleneck_S2(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S2, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3
        down_scale = 2
        low_channel = planes//down_scale


        self.point_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.pointconv2 = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize='test')







    def forward(self, x):
        identify = x
        x = self.point_conv(x)
        x = self.pointconv2(x)
        x = x.replace_feature(identify.features + x.features)
        #x = x.replace_feature(self.channel_shuffle(torch.cat([csp1.features,x.features],dim=-1),2))
        return x


class Bottleneck_S1(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S1, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3
        down_scale = 2
        low_channel = planes//down_scale


        self.point_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.sparse_conv = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion)







    def forward(self, x):
        identify = x
        x = self.point_conv(x)
        x = self.sparse_conv(x)
        x = x.replace_feature(identify.features + x.features)
        #x = x.replace_feature(self.channel_shuffle(torch.cat([csp1.features,x.features],dim=-1),2))
        return x


class Voxel2Backbone_ours(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        if model_cfg.BottleType == "S2":
            bottleneck = Bottleneck_S2
        elif model_cfg.BottleType == "S1":
            bottleneck = Bottleneck_S1
        elif model_cfg.BottleType == "S3":
            bottleneck = Bottleneck_S3

        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
        self.attentation = Point_Attentation()
        self.conv1 = spconv.SparseSequential(
            bottleneck(32, 32, norm_fn=norm_fn, indice_key='res1'),
            bottleneck(32, 32, norm_fn=norm_fn, indice_key='res1'),
            bottleneck(32, 32, norm_fn=norm_fn, indice_key='res1'),

        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            bottleneck(64, 64, norm_fn=norm_fn, indice_key='res2'),
            bottleneck(64, 64, norm_fn=norm_fn, indice_key='res2'),
            bottleneck(64, 64, norm_fn=norm_fn, indice_key='res2'),
            bottleneck(64, 64, norm_fn=norm_fn, indice_key='res2'),

        )#2x

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),
            bottleneck(128, 128, norm_fn=norm_fn, indice_key='res3'),

        )#4x

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2),
                  indice_key='spconv4', conv_type='spconv'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res4'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res4'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res4'),

        )#8x

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv5', conv_type='spconv'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res5'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res5'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res5'),

        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv6', conv_type='spconv'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res6'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res6'),
            bottleneck(256, 256, norm_fn=norm_fn, indice_key='res6'),

        )
        if self.model_cfg.FUSION == "None":
            pass
        else:
            self.fusion = fusion(256, 512, norm_fn=None, normal_dim=256)
            self.fusion2 = fusion(256, 512, norm_fn=None, normal_dim=256)
            self.shared_conv = spconv.SparseSequential(
                spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
            )
            self.shared_conv2 = spconv.SparseSequential(
                spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
            )

        self.use_attentation = model_cfg.Attentation




        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}
        frozen_weiht = model_cfg.get('Frozen_weight', False)
        if frozen_weiht:
            for param in self.parameters():
                param.requires_grad = False

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
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )# sparselizing
        if self.use_attentation:
            x_conv1 = self.attentation(self.conv1(input_sp_tensor))#[928, 1600] 1x
            x_conv2 = self.attentation(self.conv2(x_conv1))# [464, 800]   2x
            x_conv3 = self.attentation(self.conv3(x_conv2))#4x [232, 400]
            x_conv4 = self.attentation(self.conv4(x_conv3))#8X [116, 200]
            x_conv5 = self.attentation(self.conv5(x_conv4))#16X [58, 100]
            x_conv6 = self.attentation(self.conv6(x_conv5))#32x [29, 50]
        else:
            x_conv1 = self.conv1(input_sp_tensor)  # [928, 1600] 1x
            x_conv2 = self.conv2(x_conv1)  # [464, 800]   2x
            x_conv3 = self.conv3(x_conv2)  # 4x [232, 400]
            x_conv4 = self.conv4(x_conv3)  # 8X [116, 200]
            x_conv5 = self.conv5(x_conv4)  # 16X [58, 100]
            x_conv6 = self.conv6(x_conv5)  # 32x [29, 50]

        if self.model_cfg.FUSION == "None":
            pass
        elif self.model_cfg.FUSION == "Add_only":
            x_conv5.indices[:, 1:] *= 2
            x_conv6.indices[:, 1:] *= 4
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices, x_conv6.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            x_conv4 = self.bev_out(x_conv4)
        elif self.model_cfg.FUSION == "Trans_Add":
            x_conv5 = self.fusion(x_conv5, x_conv6)
            x_conv4 = self.fusion2(x_conv4,x_conv5)
            # fusion2 = self.fusion2(x_conv4, x_conv6)
            # x_conv4 = x_conv4.replace_feature(torch.cat([fusion1.features, fusion2.features], dim=1))
            x_conv4 = self.shared_conv(x_conv4)
            x_conv5.indices[:, 1:] *= 2
            x_conv6.indices[:, 1:] *= 4
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices, x_conv6.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            x_conv4 = self.bev_out(x_conv4)
        elif self.model_cfg.FUSION == "Trans":
            x_conv5 = self.fusion(x_conv5, x_conv6)
            x_conv4 = self.fusion2(x_conv4,x_conv5)
            # fusion2 = self.fusion2(x_conv4, x_conv6)
            # x_conv4 = x_conv4.replace_feature(torch.cat([fusion1.features, fusion2.features], dim=1))
            x_conv4 = self.shared_conv(x_conv4)
            x_conv5.indices[:, 1:] *= 2
            x_conv6.indices[:, 1:] *= 4
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices, x_conv6.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            x_conv4 = self.bev_out(x_conv4)
        elif self.model_cfg == "Add_Trans":
            x_conv5.indices[:, 1:] *= 2
            x_conv6.indices[:, 1:] *= 4
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices, x_conv6.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            x_conv4 = self.bev_out(x_conv4)
            x_conv5 = self.fusion(x_conv5, x_conv6)
            x_conv4 = self.fusion2(x_conv4, x_conv5)
            # fusion2 = self.fusion2(x_conv4, x_conv6)
            # x_conv4 = x_conv4.replace_feature(torch.cat([fusion1.features, fusion2.features], dim=1))
            x_conv4 = self.shared_conv(x_conv4)



        elif self.model_cfg == "F4F5":

            x_conv4 = self.fusion(x_conv4, x_conv5)

            x_conv5.indices[:, 1:] *= 2

            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            x_conv4 = self.bev_out(x_conv4)

        elif self.model_cfg == "fusion_all":

            x_conv5 = self.fusion(x_conv5, x_conv6)
            x_conv4 = self.fusion2(x_conv4,x_conv5)

            # x_conv5.indices[:, 1:] *= 2
            # x_conv6.indices[:, 1:] *= 4
            # x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            # x_conv4.indices = torch.cat(
            #     [x_conv4.indices, x_conv5.indices, x_conv6.indices])  # 是因为拼接所以采用bev_out进行独立不重复采样
            # x_conv4 = self.bev_out(x_conv4)
        elif self.model_cfg == "fusion_all_2":

            x_conv5 = self.fusion(x_conv5, x_conv6)
            x_conv5 = self.shared_conv(x_conv5)
            x_conv4 = self.fusion2(x_conv4,x_conv5)
            x_conv4 = self.shared_conv2(x_conv4)

            x_conv5.indices[:, 1:] *= 2
            x_conv6.indices[:, 1:] *= 4
            x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
            x_conv4.indices = torch.cat(
                [x_conv4.indices, x_conv5.indices, x_conv6.indices])
            x_conv4 = self.bev_out(x_conv4)



        batch_dict.update({
            'encoded_spconv_tensor': x_conv4,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

