import os
import sys
sys.path.insert(0, '/Users/adrianmolofsky/Downloads/CS229-Project/')

import torch
import numpy as np
from torch import nn
import torchmetrics
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import lightning as L
from lightning.pytorch.cli import LightningCLI
from data import dataloader
import nasdaqdatalink

nasdaqdatalink.ApiConfig.api_key ='5YAuUoytxGpvWSLKmEjA'

# logistic regression model
class LogisticRegression(L.LightningModule):
    def __init__(self, lr: float = 1e-3, lookback: int = 3):
        super().__init__()
        self.lr = lr
        self.linear = nn.Linear(lookback, 1)
        self.bce_loss = nn.BCELoss()

    def forward(self, x):
        y_hat = torch.sigmoid(self.linear(x))
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y_target = batch
        y_hat = self(x)
        y_target = y_target.view(-1, 1)
        loss = self.bce_loss(y_hat, y_target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_target = batch
        y_hat = self(x)
        y_target = y_target.view(-1, 1)
        loss = self.bce_loss(y_hat, y_target)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_target = batch
        y_hat = self(x)
        y_target = y_target.view(-1, 1)
        loss = self.bce_loss(y_hat, y_target)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class ETH(L.LightningDataModule):
    def __init__(self, batch_size: int = 16, token: str = 'ETHUSD', start_date: str = '2016-11-7', end_date: str = '2024-11-7', lookback: int = 3):
        super().__init__()
        self.batch_size = batch_size
        self.token = token
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback

    def setup(self, stage : str):
        dataset = dataloader.CryptoCurrencyDataset(self.token, self.start_date, self.end_date, self.lookback)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_split, val_split = random_split(dataset, [train_size, val_size])

        if stage == "fit":
            self.train = train_split
            self.val = val_split

        if stage == "test":
            self.test = val_split

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size)

def cli_main():
    cli = LightningCLI()

if __name__ == "__main__":
    cli_main()