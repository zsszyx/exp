from __future__ import absolute_import
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, old_label=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.old_label = old_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if not self.old_label:
            return self._get_single_item(indices)
        else:
            return self._get_single_item_old_label(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _get_single_item_old_label(self, index):
        fname, pid, oid, cid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, oid, cid, index
