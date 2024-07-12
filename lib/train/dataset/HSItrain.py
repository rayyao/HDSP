
import os
import os.path
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import jpeg4py_loader
from lib.train.admin.environment import env_settings
from lib.train.dataset.depth_utils import get_HSI_frame
import numpy as np
import torch
class HSI(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        root = env_settings().HSI_dir if root is None else root
        super().__init__('HSI', root, image_loader)
        self.sequence_list = self._get_sequence_list()
        seq_ids = list(range(0, len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()



    def get_name(self):
        return 'toolkit'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        with open(bb_anno_file) as f:
            ggt = np.loadtxt((x[:-2].replace('\t',',') for x in f), delimiter=',')
            gt=np.float32(ggt)
        return torch.tensor(gt)



    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible}

    def _get_frame_path(self, seq_path, frame_id):
        if frame_id==0:
            rgb_frame_path = os.path.join(seq_path, 'HSI-FalseColor', '{:04}.jpg'.format(frame_id + 1))
            last_frame_path = os.path.join(seq_path, 'HSI-FalseColor', '{:04}.jpg'.format(2))

        else:
            rgb_frame_path = os.path.join(seq_path, 'HSI-FalseColor', '{:04}.jpg'.format(frame_id + 1))
            last_frame_path = os.path.join(seq_path, 'HSI-FalseColor', '{:04}.jpg'.format(frame_id))


        hsi_frame_path = os.path.join(seq_path, 'HSI', '{:04}.png'.format(frame_id+1))

        return  rgb_frame_path ,hsi_frame_path,last_frame_path  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, hsi_frame_path,last_frame_path=self._get_frame_path(seq_path, frame_id)
        name = os.path.basename(seq_path)
        img=get_HSI_frame(rgb_frame_path,hsi_frame_path,last_frame_path,name)
        return img


    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_samframes(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        sam_img=[self._get_samframe(seq_path, f_id-1) for f_id in frame_ids]
        sam_image=sam_img[0]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        sam_frames = {}
        for key, value in anno.items():
            sam_frames[key]=[value[f_id-1, ...].clone() for f_id in frame_ids]
        return sam_frames,sam_image

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return frame_list, anno_frames, obj_meta
