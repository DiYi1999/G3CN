def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from data.MyDataset import *
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import time
import warnings
warnings.filterwarnings('ignore')


class MyLigDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(MyLigDataModule, self).__init__()
        self.data_name = args.data_name
        self.shuffle_flag = True
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.drop_last = False
        self.args = args
        self.ready_dataset_module()
        self.data_set = None

        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()
        self.timestamp_scaler = MinMaxScaler()

    def ready_dataset_module(self):
        if self.args.Dataset in globals():
            self.DataSet = globals()[self.args.Dataset]
        elif self.args.Dataset.split('_Dataset')[0][:-1] + '_Dataset' in globals():
            self.DataSet = globals()[self.args.Dataset.split('_Dataset')[0][:-1] + '_Dataset']
        elif self.args.Dataset.split('_Dataset')[0][:-2] + '_Dataset' in globals():
            self.DataSet = globals()[self.args.Dataset.split('_Dataset')[0][:-2] + '_Dataset']
        elif self.args.Dataset.split('_Dataset')[0] + '_Dataset_Dataset' in globals():
            self.DataSet = globals()[self.args.Dataset.split('_Dataset')[0] + '_Dataset_Dataset']
        elif 'Typical_Nonlinear_Operators' in self.args.Dataset:
            self.DataSet = globals()['MIC_simulate_Dataset']
        else:
            raise ValueError(f"Dataset {self.args.Dataset} not found in globals()")

    def train_dataloader(self):
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            flag='train',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler
        )
        print('train', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def val_dataloader(self):
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            flag='val',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler
        )
        print('val', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def test_dataloader(self):
        self.shuffle_flag = False
        args = self.args
        self.data_set = self.DataSet(
            args,
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data_name,
            flag='test',
            lag=args.lag,
            features=args.features,
            target=args.target,
            scale=args.scale,
            scaler=self.scaler,
            timestamp_scaler=self.timestamp_scaler
        )
        print('test', len(self.data_set))
        data_loader = DataLoader(
            self.data_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_flag,
            num_workers=self.num_workers,
            drop_last=self.drop_last)
        return data_loader

    def predict_dataloader(self):
        data_loader = self.test_dataloader()
        return data_loader
