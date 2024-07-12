import torch.utils.data

from lib.train.data.image_loader import jpeg4py_loader_w_failsafe


class BaseVideoDataset(torch.utils.data.Dataset):


    def __init__(self, name, root, image_loader=jpeg4py_loader_w_failsafe):

        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.sequence_list = []     # Contains the list of sequences.
        self.class_list = []

    def __len__(self):

        return self.get_num_sequences()

    def __getitem__(self, index):

        return None

    def is_video_sequence(self):

        return True

    def is_synthetic_video_dataset(self):

        return False

    def get_name(self):

        raise NotImplementedError

    def get_num_sequences(self):

        return len(self.sequence_list)

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_sequences_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False

    def get_sequence_info(self, seq_id):

        raise NotImplementedError

    def get_frames(self, seq_id, frame_ids, anno=None):

        raise NotImplementedError

