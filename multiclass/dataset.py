import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image


class FashionDataset(Dataset):

    def __init__(self, img_path, attr_file, transform=None):

        self.img_files = []
        file = open(img_path, "r")
        for line in file.readlines():
            self.img_files.append(line.strip('\n'))

        img_attr = pd.read_table(attr_file, sep='\s', header=None)
        self.img_attrs = np.array(img_attr)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        label = self.img_attrs[index]
        img = Image.open(os.path.join("../FashionDataset", img_file)).convert('RGB')
        img = self.transform(img)
        return img, label