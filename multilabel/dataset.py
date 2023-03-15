import pandas as pd
import numpy as np
import os
import collections
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

        category_1 = ['floral', 'graphic', 'striped', 'embroidered', 'pleated', 'solid', 'lattice']
        category_2 = ['long_sleeve', 'short_sleeve', 'sleeve_less']
        category_3 = ['maxi_length', 'mini_length', 'no_dress']
        category_4 = ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline']
        category_5 = ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit']
        category_6 = ['tight', 'loose', 'conventional']
        categories = [category_1, category_2, category_3, category_4, category_5, category_6]
        img_attrs = collections.defaultdict(list)
        for index, row in img_attr.iterrows():
            for i in range(6):
                category = categories[i]
                for cat in category:
                    img_attrs[cat].append(0)
                index = row[i]
                key = category[index]
                img_attrs[key].pop()
                img_attrs[key].append(1)
        img_attrs = pd.DataFrame(img_attrs)
        self.img_attrs = np.array(img_attrs)

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