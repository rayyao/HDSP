import torch.utils.data
from lib.train.data.image_loader import jpeg4py_loader


class BaseImageDataset(torch.utils.data.Dataset):


    def __init__(self, name, root, image_loader=jpeg4py_loader):

        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.image_list = []     # Contains the list of sequences.
        self.class_list = []

    def __len__(self):

        return self.get_num_images()

    def __getitem__(self, index):

        return None

    def get_name(self):

        raise NotImplementedError

    def get_num_images(self):

        return len(self.image_list)

    def has_class_info(self):
        return False

    def get_class_name(self, image_id):
        return None

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_images_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False

    def get_image_info(self, seq_id):

        raise NotImplementedError

    def get_image(self, image_id, anno=None):

        raise NotImplementedError

