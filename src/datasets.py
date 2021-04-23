import os

import torch
from torch.utils.data import Dataset

from utils import get_img

class ICDARDataset(Dataset):
    def __init__(self, df, image_root, transforms=None):
        self.df = df
        self.image_root = image_root
        self.transforms = transforms
        self.has_target = ('class' in df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_paths = os.path.join(self.image_root, row.image_id)
        image = get_img(image_paths)

        if self.transforms:
            image = self.transforms(image=image)['image']
        
        if self.has_target:
            return image, row["class"]
        else:
            return image

