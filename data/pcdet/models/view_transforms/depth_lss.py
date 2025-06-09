import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
	dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
	bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
	nx = torch.LongTensor(
		[(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
	)
	return dx, bx, nx


class DepthLSSTransform(nn.Module):
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
		self.dx = nn.Parameter(dx, requires_grad=False)
		self.bx = nn.Parameter(bx, requires_grad=False)
		self.nx = nn.Parameter(nx, requires_grad=False)

		self.C = out_channel
		self.frustum = self.create_frustum()
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
		)
		self.depthnet = nn.Sequential(
			nn.Conv2d(in_channel + 64, in_channel, 3, padding=1),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(True),
			nn.Conv2d(in_channel, in_channel, 3, padding=1),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(True),
			nn.Conv2d(in_channel, self.D + self.C, 1),
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
		iH, iW = self.image_size
		fH, fW = self.feature_size

		ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
		D, _, _ = ds.shape
		xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 像素坐标,拆分成88份
		ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 像素坐标,拆分成32份
		frustum = torch.stack((xs, ys, ds), -1)

		return nn.Parameter(frustum, requires_grad=False)

	def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):

		camera2lidar_rots = camera2lidar_rots.to(torch.float)
		camera2lidar_trans = camera2lidar_trans.to(torch.float)
		intrins = intrins.to(torch.float)
		post_rots = post_rots.to(torch.float)  # aug上的rot
		post_trans = post_trans.to(torch.float)  # aug上的移动

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
		geom_feats = geom_feats.to(torch.float)
		x = x.to(torch.float)

		B, N, D, H, W, C = x.shape
		Nprime = B * N * D * H * W

		# flatten x
		x = x.reshape(Nprime, C)

		# flatten indices
		geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
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
		geom_feats = geom_feats[kept]
		x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

		# collapse Z
		final = torch.cat(x.unbind(dim=2), 1)

		return final

	def get_cam_feats(self, x, d):
		B, N, C, fH, fW = x.shape

		d = d.view(B * N, *d.shape[2:])  # [6,1,256,704]
		x = x.view(B * N, C, fH, fW)  # batchsize和相机数量合并到一个维度

		d = self.dtransform(d)  # 6,64,32,88
		x = torch.cat([d, x], dim=1)
		x = self.depthnet(x)

		depth = x[:, : self.D].softmax(dim=1)
		x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

		x = x.view(B, N, self.C, self.D, fH, fW)
		x = x.permute(0, 1, 3, 4, 5, 2)
		return x

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
		BN, C, H, W = x.size()
		img = x.view(int(BN / 6), 6, C, H, W)  # 1,6,256,32,88.  这个是8x下采样的.

		camera_intrinsics = batch_dict['camera_intrinsics']  # [1,6,4,4] 6个相机的齐次型内参矩阵.
		camera2lidar = batch_dict['camera2lidar']  # 相机到lidar的外参 [1,6,4,4]
		img_aug_matrix = batch_dict['img_aug_matrix']  # [1,6,4,4]
		lidar_aug_matrix = batch_dict['lidar_aug_matrix']  ## [1,6,4,4]
		lidar2image = batch_dict['lidar2image']  # [1,6,4,4]

		intrins = camera_intrinsics[..., :3, :3]
		post_rots = img_aug_matrix[..., :3, :3]
		post_trans = img_aug_matrix[..., :3, 3]
		camera2lidar_rots = camera2lidar[..., :3, :3]
		camera2lidar_trans = camera2lidar[..., :3, 3]  # 以上代码去掉齐次坐标,转换为3 x3

		points = batch_dict['points']  # batchid,x,y,z,r,t

		batch_size = BN // 6  # 相机的数量，根据自己的数量修改.
		depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)  # [1,6,1,256,704]距离矩阵

		for b in range(batch_size):
			batch_mask = points[:, 0] == b  # 取对应batchid上的mask
			cur_coords = points[batch_mask][:, 1:4]  # x,y,z坐标
			cur_img_aug_matrix = img_aug_matrix[b]  # img aug矩阵，旋转，缩放，裁剪.
			cur_lidar_aug_matrix = lidar_aug_matrix[b]  # lidar aug矩阵
			cur_lidar2image = lidar2image[b]  # lidar2image，一步到位，包含外参内参

			# inverse aug
			cur_coords -= cur_lidar_aug_matrix[:3, 3]  # 平移
			cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
				cur_coords.transpose(1, 0)
			)  # 这里是要把数据增强后的点云反求回去正常的点云.
			# lidar2image
			cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
			cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)  # 这段代码的目的是将点云坐标 cur_coords 从激光雷达坐标系转换到相机坐标系
			# 转换到了图像的坐标系.

			# get 2d coords
			dist = cur_coords[:, 2, :]  # 这是深度信息了，相机坐标系的z是深度
			cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)  # 去除掉异常距离的点.
			cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]  # x = (f*x_c)/z_c, y = (f*y_c)/z_c 这里是要为转换为像素坐标做准备

			# do image aug
			cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
			cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)  # 这个img_aug_matrix需要关注一下和内参的关系.
			cur_coords = cur_coords[:, :2, :].transpose(1, 2)  # 数据增强策略，旋转和翻转的一些操作.

			# normalize coords for grid sample
			cur_coords = cur_coords[..., [1, 0]]  # 这里调换的原因是，Lidar使用的是笛卡尔坐标系！

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
		)  # 将生成的假设视锥，转换到lidar坐标系下
		# use points depth to assist the depth prediction in images
		x = self.get_cam_feats(img, depth)
		x = self.bev_pool(geom, x)
		x = self.downsample(x)
		# convert bev features from (b, c, x, y) to (b, c, y, x)
		x = x.permute(0, 1, 3, 2)
		batch_dict['spatial_features_img'] = x  # 1,80,180,180 bev空间.
		return batch_dict