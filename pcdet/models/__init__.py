from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    if 'Custom' in batch_dict:#针对于我自己的数据集.
        for key, val in batch_dict.items():
            if key == 'camera_imgs':
                batch_dict[key] = val.cuda()
            elif not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'image_paths', 'ori_shape', 'img_process_infos']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
    else:
        for key, val in batch_dict.items():
            if key == 'camera_imgs':
                batch_dict[key] = val.cuda()
            elif not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib', 'image_paths','ori_shape','img_process_infos']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        #print(batch_dict['frame_id'])
        #print(batch_dict['gt_boxes'])
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
