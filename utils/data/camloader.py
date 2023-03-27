import collections
import random
import utils.data.transforms as T
from utils.data.transforms import InterpolationMode
from PIL import Image
import os.path as osp
import torch

factory = {
    'market1501': 'Market1501',
    'msmt17': 'MSMT17',
    'personx': 'PersonX',
    'veri': 'VeRi',
    'dukemtmc': 'DukeMTMCreID'
}


class CamLoader:
    def __init__(self, fnames, cams, batch_size, root):
        self.cam_dict = collections.defaultdict(list)
        self.__build__(fnames, cams)
        self.batch_size = 64
        self.root = '/media/lab225/diskA/dataset/ReID-data' + f'/{root}' + f'/{factory[root]}'

    def __build__(self, fnames, cams):
        assert len(fnames) == len(cams)
        for index, cam in enumerate(cams):
            self.cam_dict[cam].append(fnames[index])

    def __get_files(self):
        cams = list(self.cam_dict.keys())
        if len(cams) < 2:
            return list(self.cam_dict.values())[0], list(self.cam_dict.values())[0]
        c1, c2 = random.sample(cams, 2)
        list1 = self.cam_dict[c1]
        list2 = self.cam_dict[c2]
        if len(list1) >= self.batch_size and len(list2) >= self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = min([len(list2), len(list1)])
        list1 = random.sample(list1, batch_size)
        list2 = random.sample(list2, batch_size)
        assert len(list1) == len(list2)
        return list1, list2

    def next(self):
        data1 = []
        data2 = []
        f1, f2 = self.__get_files()
        for name1, name2 in zip(f1, f2):
            name1 = osp.join(self.root, name1)
            name2 = osp.join(self.root, name2)
            img1 = Image.open(name1).convert('RGB')
            img2 = Image.open(name2).convert('RGB')
            img1 = transform(img1)
            img2 = transform(img2)
            data1.append(img1)
            data2.append(img2)
        data1 = torch.stack(data1, dim=0)
        data2 = torch.stack(data2, dim=0)
        return data1, data2


normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
transform= T.Compose([
    T.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 128)),
    T.ToTensor(),
    normalizer,
    T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
])
