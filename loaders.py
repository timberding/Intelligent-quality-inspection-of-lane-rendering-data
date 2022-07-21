# -*- coding: UTF-8 -*-
import os

import torch.utils.data as data

from PIL import Image
import csv
from itertools import islice
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ImageSet(data.Dataset):
    def __init__(
            self,
            imagedir,
            metadir,
            metafile,
            transform,
            require_label=True
    ):
        self.imagedir = imagedir
        self.transform = transform
        self.require_label = require_label

        self.images, self.labels = self.get_txt_info(metadir, metafile)

    def get_txt_info(self, metadir, metafile):
        images = list()
        labels = list()
        with open(os.path.join(metadir, metafile), 'r', encoding='utf8') as fp:
            reader = csv.reader(fp)
            for row in islice(reader, 1, None):
                imagename = row[0]
                images.append(imagename)
                if self.require_label:
                    class_label = row[1]
                    if class_label == '0':
                        label = 0
                    else:
                        label = 1
                    labels.append(label)
        if self.require_label:
            return images, labels
        else:
            return images, None

    def __getitem__(self, item):
        imagename = self.images[item]
        image = Image.open(os.path.join(self.imagedir, imagename))
        image = image.convert('RGB')
        image = self.transform(image)
        if self.require_label:
            label = self.labels[item]
            return image, label
        else:
            return image, imagename

    def __len__(self):
        return len(self.images)


def build_loader(
        imagedir,
        batch_size,
        num_workers,
        metadir,
        metafile,
        require_label=True
):
    trfs = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    trfs = transforms.Compose(trfs)
    dataset = ImageSet(imagedir, metadir, metafile, trfs, require_label)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader



