import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, utils


class DogVaeDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None) -> None:
        self.y = csv
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.y.iloc[index,0]+'.jpg')
        img = io.imread(img_name)
        img = self.transform(img)
        label = torch.Tensor(self.y.iloc[index,1])
        return (img,label)