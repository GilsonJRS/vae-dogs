import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms, utils


class DogVaeDataset(Dataset):
    def __init__(self, csv, root_dir, transform) -> None:
        self.y = pd.read_csv(csv)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.y.iloc[index,0]+'.jpg')
        img = io.imread(img_name)
        img = transforms.ToTensor(img)
        label = torch.Tensor(self.y.iloc[index,1])
        return (img,label)