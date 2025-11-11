import os
from typing import Any, Optional
import numpy as np
import pandas as pd
import time


def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

import torch
from utils.performance import *
from utils.process import *
from utils.plot import *
from data.graph_calculate import Graph_calculate
from model.ours.MySTGNN import *
from utils.MyOptimizers import *
from utils.decompose import *


class MyLigModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.training_loss = 0
        self.val_loss = []
        self.val_loss_end = 0
        self.test_time = 0

        self.MySTGNN = MySTGNN(self.args)

        self.loss = torch.nn.MSELoss()

        self.trues = []
        self.preds = []

        self.Y_orig = torch.Tensor()
        self.Y_fore = torch.Tensor()
        self.label = torch.Tensor()
        self.all_label = torch.Tensor()
        self.time_label = torch.Tensor()
        self.S = torch.Tensor()
        self.detect_01 = torch.Tensor()
        self.exam_result = {}

        self.configure_optimizers()

    def configure_optimizers(self):

        return MyOptimizers(self)

    def training_step(self, batch, batch_idx):
        A, X, Y, T = batch
        A = A[0]
        X = X.permute(0, 2, 1)
        if self.args.BaseOn == 'reconstruct':
            Y = Y.permute(0, 2, 1)
        elif self.args.BaseOn == 'forecast':
            Y = Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        T = T.permute(0, 2, 1)
        if self.args.if_timestamp == True:
            X = torch.cat((X, T), dim=1)

        H = self.MySTGNN(X, A)

        if self.args.if_timestamp == True:
            H = H[:, :-self.args.timestamp_dim, :]
        if self.args.Decompose != 'None':
            H = Reconstruct_fuc(self.args, H)
        if H.shape != Y.shape:
            raise Exception("H.shape != Y.shape, please check the model!")
        if self.args.channel_to_channel == 'M':
            loss = self.loss(H, Y)
        elif self.args.channel_to_channel == 'MS':
            loss = self.loss(H[:, self.args.MS_which, :], Y[:, self.args.MS_which, :])
        self.log(name='training_loss', value=loss, on_epoch=True, prog_bar=True, logger=True)
        self.training_loss = loss.item()

        return loss


    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            A, X, Y, T = batch
        elif len(batch) == 6:
            A, X, Y, T, _, _ = batch
        A = A[0]
        X = X.permute(0, 2, 1)
        if self.args.BaseOn == 'reconstruct':
            Y = Y.permute(0, 2, 1)
        elif self.args.BaseOn == 'forecast':
            Y = Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        T = T.permute(0, 2, 1)
        if self.args.if_timestamp == True:
            X = torch.cat((X, T), dim=1)

        H = self.MySTGNN(X, A)

        if self.args.if_timestamp == True:
            H = H[:, :-self.args.timestamp_dim, :]
        if self.args.Decompose != 'None':
            H = Reconstruct_fuc(self.args, H)
        if H.shape != Y.shape:
            raise Exception("H.shape != Y.shape, please check the model!")
        if self.args.channel_to_channel == 'M':
            loss = self.loss(H, Y)
        elif self.args.channel_to_channel == 'MS':
            loss = self.loss(H[:, self.args.MS_which, :], Y[:, self.args.MS_which, :])

        self.val_loss.append(loss)

        return {"val_loss": loss}


    def on_validation_epoch_end(self):
        self.val_loss_end = torch.stack(self.val_loss).mean()
        self.log(name='validation_epoch_loss', value=self.val_loss_end, on_epoch=True, prog_bar=True, logger=True)
        self.val_loss.clear()

        return self.val_loss_end


    def on_validation_end(self):
        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name + '_val.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**args_dict, 'val_loss_end': self.val_loss_end.item()}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_val.csv', index=False, mode='a',
                      header=True)
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**args_dict, 'val_loss_end': self.val_loss_end.item()}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_val.csv', index=False, mode='a',
                      header=False)


    def test_step(self, batch, batch_idx):
        if batch_idx == 1:
            start_time = time.perf_counter()

        if self.args.TASK == 'anomaly_detection':
            self.Y_orig, self.Y_fore, self.label, self.all_label = self.test_step_AD(batch, batch_idx)
        elif self.args.TASK == 'forecast':
            self.Y_orig, self.Y_fore, self.trues, self.preds = self.test_step_FC(batch, batch_idx)
        elif self.args.TASK == 'reconstruct':
            self.Y_orig, self.Y_fore = self.test_step_RE(batch, batch_idx)
        else:
            raise Exception("No such TASK! must be 'anomaly_detection' or 'forecast' or 'reconstruct'!")

        if batch_idx == 1:
            self.test_time = time.perf_counter() - start_time
            print('test_time:', self.test_time)

        return self.Y_orig, self.Y_fore, self.label, self.all_label if self.args.TASK == 'anomaly_detection' else self.Y_orig, self.Y_fore


    def test_step_AD(self, batch, batch_idx):
        A, X, Y, T, label, all_label = batch
        A = A[0]
        X = X.permute(0, 2, 1)
        if self.args.BaseOn == 'reconstruct':
            Y = Y.permute(0, 2, 1)
        elif self.args.BaseOn == 'forecast':
            Y = Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        T = T.permute(0, 2, 1)
        label = label[0]
        all_label = all_label[0].permute(1, 0)
        if self.args.if_timestamp == True:
            X = torch.cat((X, T), dim=1)

        H = self.MySTGNN(X, A)

        if self.args.if_timestamp == True:
            H = H[:, :-self.args.timestamp_dim, :]
        if self.args.Decompose != 'None':
            H = Reconstruct_fuc(self.args, H)
        if H.shape != Y.shape:
            raise Exception("H.shape != Y.shape, please check the model!")

        if self.args.channel_to_channel == 'MS':
            Y = Y[:, self.args.MS_which, :]
            Y = Y.unsqueeze(1)
            H = H[:, self.args.MS_which, :]
            H = H.unsqueeze(1)

        if batch_idx == 0:
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            self.label = label
            if isinstance(all_label, np.ndarray):
                all_label = torch.from_numpy(all_label)
            self.all_label = all_label

            self.Y_orig = Y[0, :, :]
            self.Y_fore = H[0, :, :]

            self.Y_orig = torch.cat((self.Y_orig, Y[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)

        return self.Y_orig, self.Y_fore, self.label, self.all_label

    def test_step_FC(self, batch, batch_idx):
        A, X, Y, T = batch
        A = A[0]
        X = X.permute(0, 2, 1)
        if self.args.BaseOn == 'reconstruct':
            Y = Y.permute(0, 2, 1)
        elif self.args.BaseOn == 'forecast':
            Y = Y.permute(0, 2, 1)[:, :, -self.args.pred_len:]
        T = T.permute(0, 2, 1)
        if self.args.if_timestamp == True:
            X = torch.cat((X, T), dim=1)

        H = self.MySTGNN(X, A)

        if self.args.if_timestamp == True:
            H = H[:, :-self.args.timestamp_dim, :]
        if self.args.Decompose != 'None':
            H = Reconstruct_fuc(self.args, H)
        if H.shape != Y.shape:
            raise Exception("H.shape != Y.shape, please check the model!")

        if self.args.channel_to_channel == 'MS':
            Y = Y[:, self.args.MS_which, :]
            Y = Y.unsqueeze(1)
            H = H[:, self.args.MS_which, :]
            H = H.unsqueeze(1)

        if batch_idx == 0:
            self.Y_orig = Y[0, :, :]
            self.Y_fore = H[0, :, :]

            self.Y_orig = torch.cat((self.Y_orig, Y[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)

        true = Y.detach().cpu().numpy()
        pred = H.detach().cpu().numpy()
        self.trues.append(true)
        self.preds.append(pred)

        return self.Y_orig, self.Y_fore, self.trues, self.preds

    def test_step_RE(self, batch, batch_idx):
        if self.args.BaseOn != 'reconstruct':
            raise Exception("if TASK is 'reconstruct', BaseOn must be 'reconstruct'!")
        if self.args.if_time_dimOne:
            A, X, Y, T, time_label = batch
        else:
            A, X, Y, T = batch
        A = A[0]
        X = X.permute(0, 2, 1)
        Y = Y.permute(0, 2, 1)
        T = T.permute(0, 2, 1)
        if self.args.if_timestamp == True:
            X = torch.cat((X, T), dim=1)

        H = self.MySTGNN(X, A)

        if self.args.if_timestamp == True:
            H = H[:, :-self.args.timestamp_dim, :]
        if self.args.Decompose != 'None':
            H = Reconstruct_fuc(self.args, H)
        if H.shape != Y.shape:
            raise Exception("H.shape != Y.shape, please check the model!")

        if self.args.channel_to_channel == 'MS':
            Y = Y[:, self.args.MS_which, :]
            Y = Y.unsqueeze(1)
            H = H[:, self.args.MS_which, :]
            H = H.unsqueeze(1)

        if batch_idx == 0:
            if self.args.if_time_dimOne:
                time_label = time_label[0]
                time_label = torch.squeeze(time_label)
                if isinstance(time_label, np.ndarray):
                    time_label = torch.from_numpy(time_label)
                self.time_label = time_label
            else:
                self.time_label = None
            self.Y_orig = Y[0, :, :]
            self.Y_fore = H[0, :, :]
            self.Y_orig = torch.cat((self.Y_orig, Y[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[1:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)
        else:
            self.Y_orig = torch.cat((self.Y_orig, Y[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(Y.shape[1], -1)), dim=1)
            self.Y_fore = torch.cat((self.Y_fore, H[:, :, -self.args.lag_step:].permute(1, 0, 2).reshape(H.shape[1], -1)), dim=1)

        return self.Y_orig, self.Y_fore

    def on_test_epoch_end(self):
        if self.args.TASK == 'anomaly_detection':
            self.on_test_epoch_end_AD(self.Y_orig, self.Y_fore, self.label)
        elif self.args.TASK == 'forecast':
            self.on_test_epoch_end_FC(self.Y_orig, self.Y_fore, self.trues, self.preds)
        elif self.args.TASK == 'reconstruct':
            self.on_test_epoch_end_RE(self.Y_orig, self.Y_fore)
        else:
            raise Exception("No such TASK! must be 'anomaly detection' or 'forecast' or 'reconstruct'!")

    def on_test_epoch_end_AD(self, Y_orig, Y_fore, label):
        if self.args.BaseOn == 'forecast':
            self.label = label[self.args.lag:]
            self.all_label = self.all_label[:, self.args.lag:]
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")
        if self.label.shape[0] != self.Y_orig.shape[1]:
            raise Exception("label.shape[0] != Y_orig.shape[1], please check the code!")

        S = self.Y_orig - self.Y_fore
        S = torch.abs(S)
        if self.args.S_moving_average_window != 1:
            S = moving_average(S, self.args.S_moving_average_window)
        self.S = S

        result_AD = performance_AD(S.cpu().numpy(), self.label.cpu().numpy())
        print(f'F1 score: {result_AD[0]}')
        print(f'accuracy: {result_AD[1]}')
        print(f'precision: {result_AD[2]}')
        print(f'recall: {result_AD[3]}')
        print(f'AUC(ROC下面积): {result_AD[4]}')
        print(f'异常检测所选取的阈值: {result_AD[5]}')

        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)

        Y_orig_normal = self.Y_orig[:, self.label == 0]
        Y_fore_normal = self.Y_fore[:, self.label == 0]
        result_FC = performance_FC( Y_orig_normal.cpu().numpy(), Y_fore_normal.cpu().numpy())
        print(f'AD_normal_MSE: {result_FC[0]}')
        print(f'AD_normal_MAE: {result_FC[1]}')

        self.exam_result = {'F1': result_AD[0], 'precision': result_AD[2], 'recall': result_AD[3],
                            'AUC': result_AD[4], 'accuracy': result_AD[1], 'threshold': result_AD[5],
                            'normal_MSE': result_FC[0], 'normal_MAE': result_FC[1], 'test_time': self.test_time}

        ray_metric = 100 / result_AD[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', result_AD[0], prog_bar=True)
        self.log('AD_precision', result_AD[2], prog_bar=True)
        self.log('AD_recall', result_AD[3], prog_bar=True)
        self.log('AD_AUC', result_AD[4], prog_bar=True)
        self.log('AD_accuracy', result_AD[1], prog_bar=True)
        self.log('AD_threshold', result_AD[5], prog_bar=True)
        self.log('AD_normal_MSE', result_FC[0], prog_bar=True)
        self.log('AD_normal_MAE', result_FC[1], prog_bar=True)

        self.log('FC_MSE', 0, prog_bar=False)
        self.log('FC_MAE', 0, prog_bar=False)
        self.log('FC_RMSE', 0, prog_bar=False)
        self.log('FC_MAPE', 0, prog_bar=False)
        self.log('FC_MSPE', 0, prog_bar=False)

        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=True)
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=False)

        self.detect_01 = torch.where(torch.tensor(S>result_AD[5]).to(S.device),
                                     torch.tensor([1]).to(S.device),
                                     torch.tensor([0]).to(S.device))
        MyPlot_AD(self.args,
                  self.Y_orig, self.Y_fore, self.label, self.all_label,
                  self.S, self.detect_01,
                  self.exam_result, args_dict)

        print('this experiment finished')

        return self.exam_result

    def on_test_epoch_end_FC(self, Y_orig, Y_fore, trues, preds):
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")

        trues = np.concatenate(self.trues, axis=0)
        trues = np.transpose(trues, (1, 0, 2))
        trues = trues.reshape(trues.shape[0], -1)
        preds = np.concatenate(self.preds, axis=0)
        preds = np.transpose(preds, (1, 0, 2))
        preds = preds.reshape(preds.shape[0], -1)

        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)
            trues = self.trainer.datamodule.data_set.my_inverse_transform(trues)
            preds = self.trainer.datamodule.data_set.my_inverse_transform(preds)

        result_FC = performance_FC(trues, preds)

        print(f'MSE: {result_FC[0]}')
        print(f'MAE: {result_FC[1]}')
        print(f'RMSE: {result_FC[2]}')
        print(f'MAPE: {result_FC[3]}')
        print(f'MSPE: {result_FC[4]}')
        print(f'RSE: {result_FC[5]}')
        print(f'CORR: {result_FC[6]}')

        self.exam_result = {'MSE': result_FC[0], 'MAE': result_FC[1], 'RMSE': result_FC[2],
                            'MAPE': result_FC[3], 'MSPE': result_FC[4], 'test_time': self.test_time,
                            'RSE': result_FC[5], 'CORR': result_FC[6]}

        ray_metric = result_FC[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', 0, prog_bar=False)
        self.log('AD_precision', 0, prog_bar=False)
        self.log('AD_recall', 0, prog_bar=False)
        self.log('AD_AUC', 0, prog_bar=False)
        self.log('AD_accuracy', 0, prog_bar=False)
        self.log('AD_threshold', 0, prog_bar=False)
        self.log('AD_normal_MSE', 0, prog_bar=False)
        self.log('AD_normal_MAE', 0, prog_bar=False)

        self.log('FC_MSE', result_FC[0], prog_bar=True)
        self.log('FC_MAE', result_FC[1], prog_bar=True)
        self.log('FC_RMSE', result_FC[2], prog_bar=True)
        self.log('FC_MAPE', result_FC[3], prog_bar=True)
        self.log('FC_MSPE', result_FC[4], prog_bar=True)
        self.log('FC_RSE', result_FC[5], prog_bar=True)
        self.log('FC_CORR', result_FC[6], prog_bar=True)

        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=True)
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=False)

        MyPlot_FC(self.args,
                  self.Y_orig, self.Y_fore,
                  self.exam_result, args_dict)

        print('this experiment finished')

        return self.exam_result

    def on_test_epoch_end_RE(self, Y_orig, Y_fore):
        if self.Y_orig.shape != self.Y_fore.shape:
            raise Exception("Y_orig.shape != Y_fore.shape, please check the code!")

        if self.args.scale and self.args.inverse:
            self.Y_orig = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_orig)
            self.Y_fore = self.trainer.datamodule.data_set.my_inverse_transform(self.Y_fore)

        result_FC = performance_FC(self.Y_orig.cpu().numpy(), self.Y_fore.cpu().numpy())
        print(f'MSE: {result_FC[0]}')
        print(f'MAE: {result_FC[1]}')
        print(f'RMSE: {result_FC[2]}')
        print(f'MAPE: {result_FC[3]}')
        print(f'MSPE: {result_FC[4]}')
        print(f'RSE: {result_FC[5]}')
        print(f'CORR: {result_FC[6]}')

        self.exam_result = {'MSE': result_FC[0], 'MAE': result_FC[1], 'RMSE': result_FC[2],
                            'MAPE': result_FC[3], 'MSPE': result_FC[4], 'test_time': self.test_time,
                            'RSE': result_FC[5], 'CORR': result_FC[6]}

        ray_metric = result_FC[0]

        self.log('train_end_loss', self.training_loss, prog_bar=True)
        self.log('ray_metric', ray_metric, prog_bar=True)

        self.log('AD_F1', 0, prog_bar=False)
        self.log('AD_precision', 0, prog_bar=False)
        self.log('AD_recall', 0, prog_bar=False)
        self.log('AD_AUC', 0, prog_bar=False)
        self.log('AD_accuracy', 0, prog_bar=False)
        self.log('AD_threshold', 0, prog_bar=False)
        self.log('AD_normal_MSE', 0, prog_bar=False)
        self.log('AD_normal_MAE', 0, prog_bar=False)

        self.log('FC_MSE', result_FC[0], prog_bar=True)
        self.log('FC_MAE', result_FC[1], prog_bar=True)
        self.log('FC_RMSE', result_FC[2], prog_bar=True)
        self.log('FC_MAPE', result_FC[3], prog_bar=True)
        self.log('FC_MSPE', result_FC[4], prog_bar=True)
        self.log('FC_RSE', result_FC[5], prog_bar=True)
        self.log('FC_CORR', result_FC[6], prog_bar=True)

        dirname_path = self.args.table_save_path + '/'
        if not os.path.exists(dirname_path + self.args.exp_name + '_test.csv'):
            os.makedirs(dirname_path, exist_ok=True)
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=True)
        else:
            args_list = vars(self.args)
            args_dict = {k: str(v) for k, v in args_list.items()}
            save_dict = {**self.exam_result, **args_dict}
            df = pd.DataFrame(save_dict, index=[0])
            df.to_csv(self.args.table_save_path + '/' + self.args.exp_name + '_test.csv', index=False, mode='a',
                      header=False)

        # 画图
        if not self.args.if_time_dimOne:
            MyPlot_RE(self.args,
                      self.Y_orig, self.Y_fore,
                      self.exam_result, args_dict)
        else:
            MyPlot_RE(self.args,
                      self.Y_orig, self.Y_fore,
                      self.exam_result, args_dict,
                      self.time_label)


        print('this experiment finished')

        return self.exam_result

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if batch_idx ==0:
            print('this experiment finished')
