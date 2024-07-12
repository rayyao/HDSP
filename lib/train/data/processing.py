import pdb

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:

    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):

        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class HDSPProcessing(BaseProcessing):


    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):


        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):

        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes stack (Ns, 4)
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box
            # Here, we normalize anno to 0-1
            crops, boxes, _, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                            data[s + '_anno'], self.search_area_factor[s],
                                                            self.output_sz[s])

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        data['valid'] = True
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
