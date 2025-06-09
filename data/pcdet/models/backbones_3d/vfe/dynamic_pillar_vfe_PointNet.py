import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
#
# #1. Pointnet and max feature
# class PointNet_plus_max(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  use_norm=True,
#                  last_layer=False):
#         super().__init__()
#         self.last_vfe = last_layer
#         self.use_norm = use_norm
#
#
#         self.linear = nn.Sequential(
#                 nn.Linear(in_channels, out_channels, bias=False),
#                 nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
#                 nn.ReLU(),
#             )
#         self.linear2 = nn.Sequential(
#             nn.Linear(out_channels, out_channels, bias=False),
#             nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
#             nn.ReLU(),
#         )
#
#     def forward(self, inputs, unq_inv):
#         x = self.linear(inputs)
#         x2 = self.linear2(x)
#         max_pool = torch.max(x2, dim=0)[0].unsqueeze(0)
#         x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
#         max_pool = max_pool.expand(x_max.size(0), -1)
#         x_max = torch.cat((x_max, max_pool), dim=1)
#         return x_max




class sum_only(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )


    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        group_sum = torch_scatter.scatter_add(x, unq_inv, dim=0)
        return group_sum

class mean_only(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )


    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        group_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
        return group_mean


# max noly
class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        out_channels = out_channels
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated



class PointNet_V3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm


        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.linear2 = nn.Sequential(
            nn.Linear(in_channels, 2*out_channels, bias=False),
            nn.BatchNorm1d(2*out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(2*out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def channel_shuffle(self,x,groups):
        num_points,num_channels = x.size()
        channels_pergropus = num_channels // groups
        x = x.view(num_points,groups,channels_pergropus)
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_points,-1)
        return x

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x2 = self.linear2(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        group_sum = torch_scatter.scatter_add(x, unq_inv, dim=0)
        max_pool = torch.max(x2, dim=0)[0].unsqueeze(0)
        max_pool = max_pool.expand(x_max.size(0), -1)
        x = torch.cat((group_sum,x_max,max_pool), dim=1)
        x = self.channel_shuffle(x,3)


        return x



class sum_plus_max(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.linear2 = nn.Sequential(
                nn.Linear(2*out_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
    def channel_shuffle(self,x,groups):
        num_points,num_channels = x.size()
        channels_pergropus = num_channels // groups
        x = x.view(num_points,groups,channels_pergropus)
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_points,-1)
        return x

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        group_sum = torch_scatter.scatter_add(x, unq_inv, dim=0)
        x = torch.cat((x_max,group_sum), dim=1)
        x = self.channel_shuffle(x,2)
        x = self.linear2(x)

        return x



class sum_plus_mean(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.linear2 = nn.Sequential(
                nn.Linear(2*out_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
    def channel_shuffle(self,x,groups):
        num_points,num_channels = x.size()
        channels_pergropus = num_channels // groups
        x = x.view(num_points,groups,channels_pergropus)
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_points,-1)
        return x

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        group_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
        x = torch.cat((x_max,group_mean), dim=1)
        x = self.channel_shuffle(x,2)
        x = self.linear2(x)

        return x

class max_plus_mean(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.linear2 = nn.Sequential(
                nn.Linear(2*out_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
    def channel_shuffle(self,x,groups):
        num_points,num_channels = x.size()
        channels_pergropus = num_channels // groups
        x = x.view(num_points,groups,channels_pergropus)
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_points,-1)
        return x

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        group_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
        x = torch.cat((x_max,group_mean), dim=1)
        x = self.channel_shuffle(x,2)
        x = self.linear2(x)

        return x

class all(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        self.linear = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        self.linear2 = nn.Sequential(
                nn.Linear(3*out_channels, 2*out_channels, bias=False),
                nn.BatchNorm1d(2*out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Linear(2*out_channels, out_channels, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
    def channel_shuffle(self,x,groups):
        num_points,num_channels = x.size()
        channels_pergropus = num_channels // groups
        x = x.view(num_points,groups,channels_pergropus)
        x = x.transpose(1, 2).contiguous()
        x = x.view(num_points,-1)
        return x

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        group_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
        group_sum = torch_scatter.scatter_sum(x, unq_inv, dim=0)
        x = torch.cat((x_max,group_mean,group_sum), dim=1)
        x = self.channel_shuffle(x, 3)
        x = x_max + self.linear2(x)


        return x








class DynamicPillarVFESimple2DMax(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE #使用距离，特征数+1
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ #使用绝对坐标，特征数+3
        #self.r_normal = self.model_cfg.R
        # self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)
        if self.use_absolute_xyz:
            num_point_features += 3
        # if self.use_cluster_xyz:
        #     num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS #应该是指的是输出的特征数量.
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) #[7，32]

        pfn_layers = []
        # Note that pointnet can not use it alone
        if self.model_cfg.MODE == "max":
            Feature_Net = PFNLayerV2 #
        elif self.model_cfg.MODE == 'sum':
            Feature_Net = sum_only  #
        elif self.model_cfg.MODE == 'mean':
            Feature_Net = mean_only #
        elif self.model_cfg.MODE == "sum_plus_max":
            Feature_Net = sum_plus_max #
        elif self.model_cfg.MODE == "sum_plus_mean":
            Feature_Net = sum_plus_mean #
        elif self.model_cfg.MODE == "max_plus_mean":
            Feature_Net = max_plus_mean #
        elif self.model_cfg.MODE == "all":
            Feature_Net = all #




        else:
            Feature_Net = PointNet_V3


        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                Feature_Net(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]#最靠边负轴的中心点坐标
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]#最靠边负轴的中心点坐标
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]#最靠边负轴的中心点坐标

        self.scale_xy = grid_size[0] * grid_size[1] #地表的网格数量
        self.scale_y = grid_size[1] #y轴上的网格数量

        self.grid_size = torch.tensor(grid_size[:2]).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] #batch_id, x,y,z,r
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int() #floor,向下取整,计算点云在voxel下的坐标.仍为原来的数量.
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask] #提取范围内有效的点云
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()#点集

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]  # id * xy 用来拆分不同的ids, x*scale_y + y  相当于是展平的一个操作,将点的坐标转为了voxel的展平坐标. 不同的id的跨度scale_xy是不同的.

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)#挑选出独立不重复的元素.
        #独立不重复的元素的张量. 索引张量:用于标记每个元素所在的位置,用于表示每个元素返回的独立不重复元素中的位置. 包含独立元素在原始张量中的出现次数.
        f_center = torch.zeros_like(points_xyz)

        """这里的目的是计算每个点相对于其所在体素中心的偏移量。self.x_offset、self.y_offset和self.z_offset分别是网格在x、y、z方向上的偏移量，
        它们的计算考虑了体素的大小和点云范围的起始坐标。这样，每个点的坐标都被转换为相对于其所在体素中心的相对坐标。"""
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset) # 和中心点的偏移量.
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        # if self.r_normal:
        #     points[:,4]=points[:,4]/255.0
        if self.use_absolute_xyz:#绝对坐标
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])#跳过绝对坐标

        # if self.use_cluster_xyz:
        #     points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        #     f_cluster = points_xyz - points_mean[unq_inv, :]
        #     features.append(f_cluster)

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)#unq_inv 是体素的编号

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        return batch_dict
