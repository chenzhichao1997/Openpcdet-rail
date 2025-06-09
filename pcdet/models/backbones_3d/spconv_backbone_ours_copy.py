from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


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
            out = self.conv1(x)
        else:
            out = replace_feature(x,self.conv1(x.features))
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))
        return out








#这部分的输入通道等于输出通道,s=1
class Bottleneck_S1(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S1, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3
        c1,c2,c3,c4 = planes//2,planes//4,planes//8,planes//8

        self.conv1 = Mobilelike(inplanes, c1, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.conv2 = Mobilelike(c1, c2, stride, norm_fn, downsample, indice_key,expansion)
        self.conv3 = Mobilelike(c2, c3, stride, norm_fn, downsample, indice_key,expansion)
        self.conv4 = Mobilelike(c3, c4, stride, norm_fn, downsample, indice_key,expansion)


    def forward(self, x):
        # identify = x
        x = self.conv1(x)
        csp1 = x.features

        x = self.conv2(x)
        csp2 = x.features

        x = self.conv3(x)
        csp3 = x.features

        x = self.conv4(x)
        x = x.replace_feature(torch.cat([csp1,csp2,csp3,x.features],dim=1))
        # x = identify +x
        return x



class Bottleneck_S2(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None, dwconv=True):
        super(Bottleneck_S2, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.inchanel = inplanes
        self.outchanels = planes
        expansion = 3


        self.conv1 = Mobilelike(inplanes, planes, stride, norm_fn, downsample, indice_key,expansion,ksize=1)
        self.conv2 = Mobilelike(planes, planes, stride, norm_fn, downsample, indice_key,expansion)




    def forward(self, x):
        identify = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = identify +x
        return x




class VoxelResBackBone8xVoxelNeXt2DV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(96, 64, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(64, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

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
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
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


class VoxelResBackBone8xVoxelNeXt2DV0(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2),
                  indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

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
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
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


class VoxelResBackBone8xVoxelNeXt2DV1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(64, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2),
                  indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

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
        )

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
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
class Voxel2Backbone_ours(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block

        spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = spconv.SparseSequential(
            Bottleneck_S2(32, 32, norm_fn=norm_fn, indice_key='res1'),
            Bottleneck_S2(32, 32, norm_fn=norm_fn, indice_key='res1'),
            Bottleneck_S2(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
                  indice_key='spconv2', conv_type='spconv'),
            Bottleneck_S2(64, 64, norm_fn=norm_fn, indice_key='res2'),
            Bottleneck_S2(64, 64, norm_fn=norm_fn, indice_key='res2'),
            Bottleneck_S2(64, 64, norm_fn=norm_fn, indice_key='res2'),
            Bottleneck_S2(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )#2x

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2),
                  indice_key='spconv3', conv_type='spconv'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
            Bottleneck_S2(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )#4x

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2),
                  indice_key='spconv4', conv_type='spconv'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res4'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res4'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )#8x

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv5', conv_type='spconv'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res5'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res5'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res5'),
        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
                  indice_key='spconv6', conv_type='spconv'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res6'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res6'),
            Bottleneck_S2(256, 256, norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }
        self.forward_ret_dict = {}

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

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

        x_conv1 = self.conv1(input_sp_tensor)#[928, 1600] 1x
        x_conv2 = self.conv2(x_conv1)# [464, 800]   2x
        x_conv3 = self.conv3(x_conv2)#4x [232, 400]
        x_conv4 = self.conv4(x_conv3)#8X [116, 200]
        x_conv5 = self.conv5(x_conv4)#16X [58, 100]
        x_conv6 = self.conv6(x_conv5)#32x [29, 50]

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
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

# class Voxel2Backbone_ours(nn.Module):
#     def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
#         self.sparse_shape = grid_size[[1, 0]]
#
#         block = post_act_block
#
#         spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])
#
#         self.conv1 = spconv.SparseSequential(
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
#             SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
#         )
#
#         self.conv2 = spconv.SparseSequential(
#             # [1600, 1408] <- [800, 704]
#             block(32, 64, spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0] // 2),
#                   indice_key='spconv2', conv_type='spconv'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#             SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
#         )
#
#         self.conv3 = spconv.SparseSequential(
#             # [800, 704] <- [400, 352]
#             block(64, 128, spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1] // 2),
#                   indice_key='spconv3', conv_type='spconv'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#             SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
#         )
#
#         self.conv4 = spconv.SparseSequential(
#             # [400, 352] <- [200, 176]
#             block(128, 256, spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2] // 2),
#                   indice_key='spconv4', conv_type='spconv'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
#         )
#
#         self.conv5 = spconv.SparseSequential(
#             # [400, 352] <- [200, 176]
#             block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
#                   indice_key='spconv5', conv_type='spconv'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5'),
#         )
#
#         self.conv6 = spconv.SparseSequential(
#             # [400, 352] <- [200, 176]
#             block(256, 256, spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3] // 2),
#                   indice_key='spconv6', conv_type='spconv'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
#             SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res6'),
#         )
#
#         self.conv_out = spconv.SparseSequential(
#             # [200, 150, 5] -> [200, 150, 2]
#             spconv.SparseConv2d(256, 256, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
#             norm_fn(256),
#             nn.ReLU(),
#         )
#
#         self.shared_conv = spconv.SparseSequential(
#             spconv.SubMConv2d(256, 256, 3, stride=1, padding=1, bias=True),
#             nn.BatchNorm1d(256),
#             nn.ReLU(True),
#         )
#
#         self.num_point_features = 256
#         self.backbone_channels = {
#             'x_conv1': 32,
#             'x_conv2': 64,
#             'x_conv3': 128,
#             'x_conv4': 256,
#             'x_conv5': 256
#         }
#         self.forward_ret_dict = {}
#
#     def bev_out(self, x_conv):
#         features_cat = x_conv.features
#         indices_cat = x_conv.indices
#
#         indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
#         features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
#         features_unique.index_add_(0, _inv, features_cat)
#
#         x_out = spconv.SparseConvTensor(
#             features=features_unique,
#             indices=indices_unique,
#             spatial_shape=x_conv.spatial_shape,
#             batch_size=x_conv.batch_size
#         )
#         return x_out
#
#     def forward(self, batch_dict):
#         pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
#         batch_size = batch_dict['batch_size']
#         input_sp_tensor = spconv.SparseConvTensor(
#             features=pillar_features,
#             indices=pillar_coords.int(),
#             spatial_shape=self.sparse_shape,
#             batch_size=batch_size
#         )# sparselizing
#
#         x_conv1 = self.conv1(input_sp_tensor)#[928, 1600] 1x
#         x_conv2 = self.conv2(x_conv1)# [464, 800]   2x
#         x_conv3 = self.conv3(x_conv2)#4x [232, 400]
#         x_conv4 = self.conv4(x_conv3)#8X [116, 200]
#         x_conv5 = self.conv5(x_conv4)#16X [58, 100]
#         x_conv6 = self.conv6(x_conv5)#32x [29, 50]
#
#         x_conv5.indices[:, 1:] *= 2
#         x_conv6.indices[:, 1:] *= 4
#         x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
#         x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])
#
#         out = self.bev_out(x_conv4)
#
#         out = self.conv_out(out)
#         out = self.shared_conv(out)
#
#         batch_dict.update({
#             'encoded_spconv_tensor': out,
#             'encoded_spconv_tensor_stride': 8
#         })
#         batch_dict.update({
#             'multi_scale_2d_features': {
#                 'x_conv1': x_conv1,
#                 'x_conv2': x_conv2,
#                 'x_conv3': x_conv3,
#                 'x_conv4': x_conv4,
#                 'x_conv5': x_conv5,
#             }
#         })
#         batch_dict.update({
#             'multi_scale_2d_strides': {
#                 'x_conv1': 1,
#                 'x_conv2': 2,
#                 'x_conv3': 4,
#                 'x_conv4': 8,
#                 'x_conv5': 16,
#             }
#         })
#
#         return batch_dict