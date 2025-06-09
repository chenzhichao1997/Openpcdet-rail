import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']#stride = 8情况下的sptial feature
        spatial_features = encoded_spconv_tensor.dense() #[2,128,2,180,180]
        N, C, D, H, W = spatial_features.shape #batchsize,C维特征,Depth,HW
        spatial_features = spatial_features.view(N, C * D, H, W)#拍扁到bev特征.和pointpillars类似.  [2,256,180,180],这个已经转换为了bev空间.和图像的形式安全一致.
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
