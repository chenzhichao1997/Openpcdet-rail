from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)  # Residual connection
        src = self.norm1(src)  # Layer normalization

        # Feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)  # Residual connection
        src = self.norm2(src)  # Layer normalization

        return src





#这部分的输入通道等于输出通道,s=1
class Transformer_Encoder(nn.Module):
    def __init__(self, model_cfg,input_channels):
        super(Transformer_Encoder, self).__init__()
        cfg_list = model_cfg.get('Transformer_cfg', None)
        self.transofrmer_encoder = nn.Sequential()
        for cfg in cfg_list:
            self.transofrmer_encoder.append(TransformerEncoderLayer(*cfg))

        self.num_bev_features = 512
    def split_tensor(self,x,indices_cat):
        batch_index = indices_cat[:, 0]
        unique_elements, counts = torch.unique(batch_index, return_counts=True)
        split_tensors = torch.split(x, tuple(counts.tolist()))
        return split_tensors


    def forward(self, batch_dict):
        x_feature =batch_dict['encoded_spconv_tensor'].features
        x_index = batch_dict['encoded_spconv_tensor'].indices
        x_feature = self.split_tensor(x_feature,x_index)
        out = []
        for feature in x_feature:
            out.append(self.transofrmer_encoder(feature))
        out = torch.cat(out, dim=0)
        batch_dict['encoded_spconv_tensor'] = batch_dict['encoded_spconv_tensor'].replace_feature(out)
        return batch_dict
























