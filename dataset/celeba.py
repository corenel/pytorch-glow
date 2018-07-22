import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from misc import util


class CelebA(Dataset):
    def __init__(self, root,
                 image_dir='images',
                 anno_file='list_attr_celeba.txt',
                 transform=None):
        """
        CelebA dataset

        :param root: path to dataset root
        :type root: str
        :param anno_file: fielname of annotation file
        :type anno_file: str
        :param transform: desired transformation for image
        """
        super().__init__()
        assert os.path.isdir(root), 'Dataset dirctory not exists: {}'.format(root)
        self.root = root
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.transform = transform

        self.data, self.attrs = self.parse_anno_file()

    def parse_anno_file(self):
        """
        Parse annotation file

        :return: image data and attributes
        :rtype: dict, list
        """
        if os.path.exists(self.anno_file):
            anno_path = self.anno_file
        elif os.path.exists(os.path.join(self.root, self.anno_file)):
            anno_path = os.path.join(self.root, self.anno_file)
        else:
            raise FileNotFoundError('Annotation file of dataset not exists: {}'.format(self.anno_file))

        data = []
        attrs = None
        num_images = 0
        with open(anno_path, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if idx == 0:
                    num_images = int(line)
                elif idx == 1:
                    attrs = line.split(' ')
                else:
                    elements = [e for e in line.split(' ') if e]
                    image_path = os.path.join(self.root, self.image_dir, elements[0])
                    image_attr = elements[1:]
                    if not os.path.exists(image_path) or not util.is_image(image_path):
                        continue
                    # 0 for -1 and 1 for 1
                    image_onehot = [int(int(attr)) for attr in image_attr]
                    data.append({
                        'path': image_path,
                        'attr': image_onehot
                    })
        print('CelebA dataset: Expect {} images with {} attributes.'.format(num_images, len(attrs)))
        print('CelebA dataset: Find {} images with {} attributes.'.format(len(data), len(data[-1]['attr'])))

        return data, attrs

    def __getitem__(self, index):
        data = self.data[index]
        image_path = data['path']
        image_attr = data['attr']

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return {
            'x': image,
            'y_onehot': np.array(image_attr, dtype='float32')
        }

    def __len__(self):
        return len(self.data)
