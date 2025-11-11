import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.process import *
import warnings
from utils.decompose import *
from data.graph_calculate import *
from utils.decompose import *

warnings.filterwarnings('ignore')


def Decompose_Update(data, flag, args):
    """
    本函数的存在目的是，如果args.Decompose不是None，就要对data进行分解，否则直接返回X；
    具体的做法和需要注意的点是：直接对data进行分解是不合适的，这放在test的数据集上那就是信息泄露；
    所以采用的方法是，要对每个data进行滑窗，滑出来的样本挨个进行分解，最后拼接起来，命名data_Decompose_stack输出；
    但需要注意的是，如果args.Decompose是None，那么在__getitem__中取样应该是滑窗，如果不是None，那么就是切割取样，这个要注意；

    Args:
        data: array(data_len, sensor_num)
        args:

    Returns:
        data_Decompose_stack:
        data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),不分解则和data一样array(data_len, sensor_num)
    """
    # 记录文件名字
    file_name = str(flag) + "_" + str(args.Dataset) + str(args.dataset_split_ratio) + "_" \
                + str(args.lag) + str(args.lag_step) + str(args.label_len)
    if args.Decompose == 'STL':
        De_info = "_" + args.Decompose + str(args.STL_seasonal)
        file_name = file_name + De_info
    else:
        De_info = "_" + args.Decompose + str(args.Wavelet_wave) + str(args.Wavelet_level)
        file_name = file_name + De_info
    if args.preMA:
        other_info = "_scale" + str(args.scale) + "_preMA" + str(args.preMA_win)
        file_name = file_name + other_info
    else:
        other_info = "_scale" + str(args.scale) + "_preMA" + str(args.preMA)
        file_name = file_name + other_info
    # file_name = "Decompose_data_stack_" + file_name + ".csv"
    file_name = "Decompose_data_stack_" + file_name + ".pkl"
    csv_dir = args.table_save_path + '/' + file_name

    if args.Decompose != 'None':
        if os.path.exists(csv_dir):
            # data_Decompose_stack = pd.read_csv(csv_dir, header=None).values
            with open(csv_dir, 'rb') as f:
                data_Decompose_stack = pickle.load(f)
        else:
            if args.BaseOn == 'forecast':
                sample_num = (data.shape[0] - args.lag - args.pred_len + args.lag_step) // args.lag_step
            elif args.BaseOn == 'reconstruct':
                sample_num = (data.shape[0] - args.lag + args.lag_step) // args.lag_step

            for index in range(sample_num):
                s_begin = index * args.lag_step
                s_end = s_begin + args.lag
                sample = data[s_begin:s_end]
                sample = torch.from_numpy(sample.T).float()
                'sample: tensor(sensor_num, lag)'  # 先转置并转化为tensor
                sample = torch.unsqueeze(sample, dim=0)
                'sample: tensor(1, sensor_num, lag)'  # 前面加个维度使其符合Decompose_fuc的输入要求
                if index == 0:
                    data_stack = sample
                else:
                    data_stack = torch.cat((data_stack, sample), dim=0)
                    'data_stack: tensor(sample_num, sensor_num, lag)'
            data_Decompose_stack = Decompose_fuc(data_stack, args)
            'data_Decompose_stack: tensor(sample_num, sensor_num*3, lag)'
            data_Decompose_stack = data_Decompose_stack.permute(1, 0, 2)
            'data_Decompose_stack: tensor(sensor_num*3, sample_num, lag)'
            data_Decompose_stack = data_Decompose_stack.reshape(data_Decompose_stack.shape[0], -1)
            'data_Decompose_stack: tensor(sensor_num*3, sample_num*lag)'
            data_Decompose_stack = data_Decompose_stack.numpy().T
            'data_Decompose_stack: array(sample_num*lag, sensor_num*3)'

            Path(os.path.dirname(csv_dir)).mkdir(parents=True, exist_ok=True)
            # pd.DataFrame(data_Decompose_stack).to_csv(csv_dir, header=None, index=None)
            with open(csv_dir, 'wb') as f:
                pickle.dump(data_Decompose_stack, f)
    else:
        data_Decompose_stack = data

    return data_Decompose_stack




class ETTh1_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***数据划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//8), train_len]
        # border2s = [train_len-(train_len//8), train_len, len(data)]
        border1s = [0, 12 * 30 * 24 - self.args.lag, 12 * 30 * 24 + 4 * 30 * 24 - self.args.lag]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class ETTh2_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh2/ETTh2', data_name='ETTh2',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***数据划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//8), train_len]
        # border2s = [train_len-(train_len//8), train_len, len(data)]
        border1s = [0, 12 * 30 * 24 - self.args.lag, 12 * 30 * 24 + 4 * 30 * 24 - self.args.lag]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class ETTm1_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTm1/ETTm1', data_name='ETTm1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it. now data_shape: {}, x_dim: {}'.format(data.shape, x_dim))

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***数据划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//8), train_len]
        # border2s = [train_len-(train_len//8), train_len, len(data)]
        border1s = [0, 12 * 30 * 24 * 4 - self.args.lag, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.args.lag]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class ETTm2_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTm2/ETTm2', data_name='ETTm2',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***数据划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//8), train_len]
        # border2s = [train_len-(train_len//8), train_len, len(data)]
        border1s = [0, 12 * 30 * 24 * 4 - self.args.lag, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.args.lag]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class weather_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='weather/weather', data_name='weather',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//8), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class Electricity_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='Electricity/Electricity', data_name='Electricity',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
        data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//8), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class exchange_rate_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='exchange_rate/exchange_rate', data_name='exchange_rate',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
        data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//8), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class traffic_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='traffic/traffic', data_name='traffic',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data_df = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data_df[[self.target]].values
            else:
                data = data_df.drop(['date'], axis=1).values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        df_stamp = data_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], axis=1).values

        """***划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//8), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class solar_energy_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='solar_energy/solar_energy', data_name='solar_energy',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        try:
            # f = open(os.path.join(self.root_path, self.data_path, '{}.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        except (KeyError, FileNotFoundError):
            data = None

        """检查数据维度是否正确"""
        x_dim = self.get_data_dim(self.data_name)
        if self.features != 'S':
            if data.shape[1] != x_dim:
                raise ValueError('Data shape error, please check it')

        """df_stamp是时间标签信息"""
        # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
        data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        train_len = int(self.args.dataset_split_ratio * len(data))
        border1s = [0, train_len-(train_len//8), train_len]
        border2s = [train_len-(train_len//8), train_len, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        data = data[border1:border2]
        data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A


    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]

        return (self.A.astype(np.float32),
                x_batch.astype(np.float32),
                y_batch.astype(np.float32),
                datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class MIC_simulate_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'train':
            # f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        elif self.flag == 'val':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_val.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_val.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None
        elif self.flag == 'test':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None

        # """检查数据维度是否正确"""
        # x_dim = self.get_data_dim(self.data_name)
        # if self.features != 'S':
        #     if data.shape[1] != x_dim:
        #         raise ValueError('Data shape error, please check it')

        """data_stamp是时间标签信息"""
        if self.args.if_time_dimOne:
            data, data_stamp = data[:, 1:], data[:, 0]
            """这个数据集第一维是t时间维（被打乱），后面几维是正常数据"""
        else:
            # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
            data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//4), train_len]
        # border2s = [train_len-(train_len//4), train_len, len(data)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # data = data[border1:border2]
        # data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]
        A = self.A

        # 如果是half_to_half，那么这个数据集就是前半部分是x，后半部分是y，用前面维度重建后面维度，而不是对这个数据集进行全体传感器互相重建
        if self.args.reco_form == 'half_to_half':
            x_batch = x_batch[:, :self.args.sensor_num]
            y_batch = y_batch[:, self.args.sensor_num:]
            A = A[:self.args.sensor_num, self.args.sensor_num:] if not self.args.if_timestamp \
                else A[:self.args.sensor_num, self.args.sensor_num:-self.data_stamp.shape[1]]

        # 如果if_time_dimOne为True，就代表这个数据集的排列不是时间递增的，一开始时间就是被打乱的
        # ，而且其文件第一列是t列，我们需要把这一列也回传，用来画图，作为横轴
        if self.flag == 'test' and self.args.if_time_dimOne:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    self.data_stamp.astype(np.float32))
        else:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1
        # if self.args.BaseOn == 'reconstruct':
        #     return len(self.data) // self.args.lag
        # elif self.args.BaseOn == 'forecast':
        #     return (len(self.data) - self.args.pred_len) // self.args.lag

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output


class Physical_System_Synthetic_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'train':
            # f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        elif self.flag == 'val':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_val.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_val.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None
        elif self.flag == 'test':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None

        # """检查数据维度是否正确"""
        # x_dim = self.get_data_dim(self.data_name)
        # if self.features != 'S':
        #     if data.shape[1] != x_dim:
        #         raise ValueError('Data shape error, please check it')

        """data_stamp是时间标签信息"""
        if self.args.if_time_dimOne:
            data, data_stamp = data[:, 1:], data[:, 0]
            """这个数据集第一维是t时间维（被打乱），后面几维是正常数据"""
        else:
            # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
            data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//4), train_len]
        # border2s = [train_len-(train_len//4), train_len, len(data)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # data = data[border1:border2]
        # data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]
        A = self.A

        # 如果是half_to_half，那么这个数据集就是前半部分是x，后半部分是y，用前面维度重建后面维度，而不是对这个数据集进行全体传感器互相重建
        if self.args.reco_form == 'half_to_half':
            x_batch = x_batch[:, :self.args.sensor_num]
            y_batch = y_batch[:, self.args.sensor_num:]
            A = A[:self.args.sensor_num, self.args.sensor_num:] if not self.args.if_timestamp \
                else A[:self.args.sensor_num, self.args.sensor_num:-self.data_stamp.shape[1]]

        # 如果if_time_dimOne为True，就代表这个数据集的排列不是时间递增的，一开始时间就是被打乱的
        # ，而且其文件第一列是t列，我们需要把这一列也回传，用来画图，作为横轴
        if self.flag == 'test' and self.args.if_time_dimOne:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    self.data_stamp.astype(np.float32))
        else:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1
        # if self.args.BaseOn == 'reconstruct':
        #     return len(self.data) // self.args.lag
        # elif self.args.BaseOn == 'forecast':
        #     return (len(self.data) - self.args.pred_len) // self.args.lag

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output



class SixD_Hyperchaotic_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'train':
            # f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        elif self.flag == 'val':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_val.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_val.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None
        elif self.flag == 'test':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None

        # """检查数据维度是否正确"""
        # x_dim = self.get_data_dim(self.data_name)
        # if self.features != 'S':
        #     if data.shape[1] != x_dim:
        #         raise ValueError('Data shape error, please check it')

        """data_stamp是时间标签信息"""
        if self.args.if_time_dimOne:
            data, data_stamp = data[:, 1:], data[:, 0]
            """这个数据集第一维是t时间维（被打乱），后面几维是正常数据"""
        else:
            # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
            data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//4), train_len]
        # border2s = [train_len-(train_len//4), train_len, len(data)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # data = data[border1:border2]
        # data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]
        A = self.A

        # 如果是half_to_half，那么这个数据集就是前半部分是x，后半部分是y，用前面维度重建后面维度，而不是对这个数据集进行全体传感器互相重建
        if self.args.reco_form == 'half_to_half':
            x_batch = x_batch[:, :self.args.sensor_num]
            y_batch = y_batch[:, self.args.sensor_num:]
            A = A[:self.args.sensor_num, self.args.sensor_num:] if not self.args.if_timestamp \
                else A[:self.args.sensor_num, self.args.sensor_num:-self.data_stamp.shape[1]]

        # 如果if_time_dimOne为True，就代表这个数据集的排列不是时间递增的，一开始时间就是被打乱的
        # ，而且其文件第一列是t列，我们需要把这一列也回传，用来画图，作为横轴
        if self.flag == 'test' and self.args.if_time_dimOne:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    self.data_stamp.astype(np.float32))
        else:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1
        # if self.args.BaseOn == 'reconstruct':
        #     return len(self.data) // self.args.lag
        # elif self.args.BaseOn == 'forecast':
        #     return (len(self.data) - self.args.pred_len) // self.args.lag

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output



class Cart_Pendulum_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'train':
            # f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        elif self.flag == 'val':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_val.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_val.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None
        elif self.flag == 'test':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None

        # """检查数据维度是否正确"""
        # x_dim = self.get_data_dim(self.data_name)
        # if self.features != 'S':
        #     if data.shape[1] != x_dim:
        #         raise ValueError('Data shape error, please check it')

        """data_stamp是时间标签信息"""
        if self.args.if_time_dimOne:
            data, data_stamp = data[:, 1:], data[:, 0]
            """这个数据集第一维是t时间维（被打乱），后面几维是正常数据"""
        else:
            # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
            data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//4), train_len]
        # border2s = [train_len-(train_len//4), train_len, len(data)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # data = data[border1:border2]
        # data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]
        A = self.A

        # 如果是half_to_half，那么这个数据集就是前半部分是x，后半部分是y，用前面维度重建后面维度，而不是对这个数据集进行全体传感器互相重建
        if self.args.reco_form == 'half_to_half':
            x_batch = x_batch[:, :self.args.sensor_num]
            y_batch = y_batch[:, self.args.sensor_num:]
            A = A[:self.args.sensor_num, self.args.sensor_num:] if not self.args.if_timestamp \
                else A[:self.args.sensor_num, self.args.sensor_num:-self.data_stamp.shape[1]]

        # 如果if_time_dimOne为True，就代表这个数据集的排列不是时间递增的，一开始时间就是被打乱的
        # ，而且其文件第一列是t列，我们需要把这一列也回传，用来画图，作为横轴
        if self.flag == 'test' and self.args.if_time_dimOne:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    self.data_stamp.astype(np.float32))
        else:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32))

    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1
        # if self.args.BaseOn == 'reconstruct':
        #     return len(self.data) // self.args.lag
        # elif self.args.BaseOn == 'forecast':
        #     return (len(self.data) - self.args.pred_len) // self.args.lag

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output




class Super_Nonlinear_Dataset_Dataset(Dataset):
    def __init__(self, args, root_path='/home/data/DiYi/DATA/forecast', flag='train', lag=None,
                 features='M', data_path='ETTh1/ETTh1', data_name='ETTh1',
                 missing_rate=0, missvalue=np.nan, target='OT', scale=False, scaler=None, timestamp_scaler=None):
        # size [seq_len, label_len pred_len]
        # info
        self.args = args
        self.lag = lag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.timestamp_scaler = timestamp_scaler

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = data_name

        self.A = None

        self.__read_data__()

    def get_data_dim(self, dataset):
        return self.args.sensor_num

    def __read_data__(self):
        """
        get data from pkl files

        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        """

        """导入数据"""
        if self.flag == 'train':
            # f = open(os.path.join(self.root_path, self.data_path, '{}_train.pkl'.format(self.data_name)), "rb")
            # data = pickle.load(f).values.reshape((-1, x_dim))
            # f.close()
            data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_name)),
                               sep=',', index_col=False)
            if self.features == 'S':
                data = data[:, [self.target]]
            data = data.values
        elif self.flag == 'val':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_val.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_val.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None
        elif self.flag == 'test':
            try:
                # f = open(os.path.join(self.root_path, self.data_path, '{}_test.pkl'.format(self.data_name)), "rb")
                # data = pickle.load(f).values.reshape((-1, x_dim))
                # f.close()
                data = pd.read_csv(os.path.join(self.root_path, self.data_path, '{}_test.csv'.format(self.data_name)),
                                   sep=',', index_col=False)
                if self.features == 'S':
                    data = data[:, [self.target]]
                data = data.values
            except (KeyError, FileNotFoundError):
                data = None

        # """检查数据维度是否正确"""
        # x_dim = self.get_data_dim(self.data_name)
        # if self.features != 'S':
        #     if data.shape[1] != x_dim:
        #         raise ValueError('Data shape error, please check it')

        """data_stamp是时间标签信息"""
        if self.args.if_time_dimOne:
            data, data_stamp = data[:, 1:], data[:, 0]
            """这个数据集第一维是t时间维（被打乱），后面几维是正常数据"""
        else:
            # 之前的的弃用了，data_stamp直接创建一个data行数长度的递增numpy序列
            data_stamp = np.arange(len(data))
        # 变成二维，后面加维度
        data_stamp = np.expand_dims(data_stamp, axis=1)

        """***划分***"""
        # train_len = int(self.args.dataset_split_ratio * len(data))
        # border1s = [0, train_len-(train_len//4), train_len]
        # border2s = [train_len-(train_len//4), train_len, len(data)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        # data = data[border1:border2]
        # data_stamp = data_stamp[border1:border2]

        """***preprocessing***"""
        """数据标准化归一化"""
        if self.scale:
            data = self.normalize(data, self.flag)
            data_stamp = self.normalize(data_stamp, self.flag, if_timestamp=True)

        """nan填充:用前一个或者后一个时间步进行nan填充"""
        if np.isnan(data).any():
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            data = df.values
            print('Data contains null values. Will be replaced with 0')

        """含噪数据滑动平均预处理"""
        if self.args.preMA:
            data = preMA(data, self.args.preMA_win)

        """如设置了分解，则进行分解"""
        data_Decompose_stack = Decompose_Update(data, self.flag, self.args)
        'data_Decompose_stack: 若分解array(lag*sample_num, sensor_num*3),' \
        '不分解则和data一样array(data_len, sensor_num)'

        """数据定型"""
        self.data = data
        self.data_Decompose_stack = data_Decompose_stack
        self.data_stamp = data_stamp

        """截取一小段数据用于计算邻接矩阵"""
        if self.args.graph_ca_len < data.shape[0]:
            data_graph_ca = data[:self.args.graph_ca_len, :]
            'data_graph_ca: array(graph_ca_len, sensor_num)'
        else:
            data_graph_ca = data
            'data_graph_ca: array(data_len, sensor_num)'
        if self.args.Decompose != 'None':
            data_graph_ca = torch.from_numpy(data_graph_ca.T).float()
            'data_graph_ca: tensor(sensor_num, graph_ca_len)'
            data_graph_ca = torch.unsqueeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(1, sensor_num, graph_ca_len)'
            data_graph_ca = Decompose_fuc(data_graph_ca, self.args)
            'data_graph_ca: tensor(1, sensor_num*3, graph_ca_len)'
            data_graph_ca = torch.squeeze(data_graph_ca, dim=0)
            'data_graph_ca: tensor(sensor_num*3, graph_ca_len)'
            data_graph_ca = data_graph_ca.numpy().T
            'data_graph_ca: array(graph_ca_len, sensor_num*3)'
        if self.args.if_timestamp:
            data_graph_ca = np.concatenate((data_graph_ca, data_stamp[:self.args.graph_ca_len, :]), axis=1)
            'data_graph_ca: array(graph_ca_len, sensor_num*3+1)'
        # 计算邻接矩阵
        if self.args.graph_if_norm_A:
            A, A_self, A_w, A_norm, A_self_norm = Graph_calculate(self.args, data_graph_ca, if_return_norm=True, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self_norm
            else:
                self.A = A_norm
        else:
            A, A_self, A_w = Graph_calculate(self.args, data_graph_ca, if_return_norm=False, flag=self.flag)
            if self.args.self_edge:
                self.A = A_self
            else:
                self.A = A

    def __getitem__(self, index):
        if self.args.Decompose != 'None':
            # 如果是被分解的，那么就是对data_Decompose_stack进行取样，而且不是滑窗取样，而是切割取样
            s_begin = index * self.lag
            s_end = s_begin + self.lag
            x_batch = self.data_Decompose_stack[s_begin:s_end]
            # 但是后续y_batch和datetime_batch还是要从原始数据中 滑窗取的，所以得在纠正回来
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
        else:
            # 如果不是被分解的，那么就是对data进行取样，而且是滑窗取样
            s_begin = index * self.args.lag_step
            s_end = s_begin + self.lag
            x_batch = self.data[s_begin:s_end]

        if self.args.BaseOn == 'reconstruct':
            r_begin = s_begin
            r_end = s_end
        elif self.args.BaseOn == 'forecast':
            r_begin = s_end - self.args.label_len
            r_end = s_end + self.args.pred_len
        y_batch = self.data[r_begin:r_end]
        datetime_batch = self.data_stamp[s_begin:s_end]
        A = self.A

        # 如果是half_to_half，那么这个数据集就是前半部分是x，后半部分是y，用前面维度重建后面维度，而不是对这个数据集进行全体传感器互相重建
        if self.args.reco_form == 'half_to_half':
            x_batch = x_batch[:, :self.args.sensor_num]
            y_batch = y_batch[:, self.args.sensor_num:]
            A = A[:self.args.sensor_num, self.args.sensor_num:] if not self.args.if_timestamp \
                else A[:self.args.sensor_num, self.args.sensor_num:-self.data_stamp.shape[1]]

        # 如果if_time_dimOne为True，就代表这个数据集的排列不是时间递增的，一开始时间就是被打乱的
        # ，而且其文件第一列是t列，我们需要把这一列也回传，用来画图，作为横轴
        if self.flag == 'test' and self.args.if_time_dimOne:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32),
                    self.data_stamp.astype(np.float32))
        else:
            return (A.astype(np.float32),
                    x_batch.astype(np.float32),
                    y_batch.astype(np.float32),
                    datetime_batch.astype(np.float32))


    def __len__(self):
        if self.args.BaseOn == 'reconstruct':
            return (len(self.data) - self.lag) // self.args.lag_step + 1
        elif self.args.BaseOn == 'forecast':
            return (len(self.data) - self.lag - self.args.pred_len) // self.args.lag_step + 1
        # if self.args.BaseOn == 'reconstruct':
        #     return len(self.data) // self.args.lag
        # elif self.args.BaseOn == 'forecast':
        #     return (len(self.data) - self.args.pred_len) // self.args.lag

    def normalize(self, data, flag, if_timestamp=False):
        """
        returns normalized and standardized data.

        data : array-like
        flag: 'train' or 'val' or 'test'
        if_timestamp: 这是对正常数据进行标准化还是对时间戳进行标准化，这决定我们调用self.scaler还是self.timestamp_scaler
        """
        if not if_timestamp:
            if flag == 'train':
                self.scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.scaler, 'scale_'):
                    self.scaler.fit(data)
            else:
                pass
            data = self.scaler.transform(data)
        else:
            if flag == 'train':
                self.timestamp_scaler.fit(data)
            elif flag in ['val', 'test']:
                # 验证时，先判断self.timestamp_scaler是否已经fit过，没有的话就fit，有就不用了
                if not hasattr(self.timestamp_scaler, 'scale_'):
                    self.timestamp_scaler.fit(data)
            else:
                pass
            data = self.timestamp_scaler.transform(data)


        return data

    def my_inverse_transform(self, data, if_timestamp=False):
        """调用此函数时一定注意此函数会打断梯度"""
        if type(data) == torch.Tensor:
            output = data.numpy()
        else:
            output = data
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 开始逆归一化/标准化
        if not if_timestamp:
            output = self.scaler.inverse_transform(output)
        else:
            output = self.timestamp_scaler.inverse_transform(output)
        # 转回data原来的形状
        if data.shape[0] < data.shape[1]:
            output = output.T
        # 转回tensor
        if type(data) == torch.Tensor:
            output = torch.from_numpy(output).float()
        return output






# if __name__ == '__main__':
#     flag = 'test'
#     dataset = NASA_Anomaly(root_path='./data/', data_path='MSL', flag=flag, size=(60, 30, 1))
#     print(flag, len(dataset))
    # data_loader = DataLoader(
    #         dataset,
    #         batch_size=32,
    #         shuffle=True,
    #         num_workers=2,
    #         drop_last=True)
    # for (x, y, x_stamp, y_stamp, label) in data_loader:
    #     print(x.size(), y.size(), x_stamp.size(), y_stamp.size(), label.size())
    #     break