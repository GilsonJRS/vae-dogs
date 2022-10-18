from asyncio.log import logger
import pandas as pd
from dataset import DogVaeDataset
from model import VanillaVAE
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger

wandb.login()

wandb_logger = WandbLogger(project='dog-vae')

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((48,48))
])

labels = pd.read_csv("./data/labels.csv")
labels['breed'] = labels['breed'].astype('category')
labels['breed'] = labels['breed'].cat.codes


dataset = DogVaeDataset(labels, './data/train/', transform=train_transforms)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [8177, 2045])

train_dataloader = DataLoader(train_dataset, num_workers=4)
val_dataloader = DataLoader(val_dataset, num_workers=4)

model = VanillaVAE(input_height=50)
trainer = pl.Trainer(enable_checkpointing=True, max_epochs=100,
devices=1, accelerator='gpu', logger=wandb_logger)

trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)