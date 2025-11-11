import argparse

import numpy as np
import torch
import pandas as pd


def moving_average(anomaly_score_tensor, window_num):
    """
    对异常评分的每一列分别进行移动平均，平滑异常评分曲线

    Args:
        anomaly_score_tensor: (all_len, node_num)异常打分
        window_num: 滑动平均窗长

    Returns:

    """
    if anomaly_score_tensor.shape[0] < anomaly_score_tensor.shape[1]:
        anomaly_score_array = anomaly_score_tensor.cpu().numpy()
        for i in range(anomaly_score_tensor.shape[0]):
            anomaly_score_array[i, :] = np.convolve(anomaly_score_array[i, :], np.ones(window_num) / window_num, mode='same')
        normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)
    else:
        anomaly_score_array = anomaly_score_tensor.cpu().numpy()
        for i in range(anomaly_score_tensor.shape[1]):
            anomaly_score_array[:, i] = np.convolve(anomaly_score_array[:, i], np.ones(window_num) / window_num, mode='same')
        normal_anomaly_score_tensor = torch.from_numpy(anomaly_score_array).float().to(anomaly_score_tensor.device)

    return normal_anomaly_score_tensor



def preIDW(data):
    """
    用前方最近的观测值填补缺失值

    Args:
        data: (all_len, node_num)，numpy数组

    Returns:

    """
    df_data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
    df_data = df_data.fillna(method="ffill", axis=0)
    df_data = df_data.fillna(method="backfill", axis=0)
    data = df_data.values if isinstance(data, np.ndarray) else df_data

    return data


# # 用前方最近的观测值填补缺失值，这个是用在batch上的，针对的是(batch_size, node_num, lag)的torch tensor，弃用了
# def preIDW(x_batch):
#     x_data = x_batch.cpu().numpy()
#     # (batch_size, node_num, lag)
#     for i in range(x_data.shape[0]):
#         i_batch = pd.DataFrame(x_data[i])
#         # (node_num, lag)
#         i_batch = i_batch.fillna(method="ffill", axis=1)
#         i_batch = i_batch.fillna(method="backfill", axis=1)
#         x_data[i] = i_batch.values
#     return torch.FloatTensor(x_data).type_as(x_batch)




def preMA(data_array, window_size=50):
    """
    Args:
        data_array: (lag, node_num)，numpy数组
        window_size:

    Returns:

    """
    # data_array = data.values
    result = np.copy(data_array)
    # 遍历每一列
    for i in range(data_array.shape[1]):
        series = pd.Series(data_array[:, i])
        smoothed = series.rolling(window=window_size, center=True).mean()
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')
        result[:, i] = smoothed.values
    # data_out = pd.DataFrame(data_array, columns=data.columns)

    return result




def make_missing_data(data, missing_rate, missvalue, norm_data=None):
    """
    生成缺失数据

    Args:
        data: (all_len, node_num), numpy数组
        missing_rate: 缺失率
        missvalue: 缺失值
        norm_data: (all_len, node_num), 归一化后的数据，可以不输入

    Returns:

    """
    missing_data = data.copy() if isinstance(data, np.ndarray) else data.values.copy()
    missing_num = int(missing_data.size * missing_rate)
    missing_position = np.random.choice(missing_data.size, missing_num, replace=False)
    missing_position_2d = np.unravel_index(missing_position, missing_data.shape)
    missing_data[missing_position_2d] = missvalue
    missing_data = pd.DataFrame(missing_data, columns=data.columns) if isinstance(data, pd.DataFrame) else missing_data

    if norm_data is not None:
        missing_norm_data = norm_data.copy() if isinstance(norm_data, np.ndarray) else norm_data.values.copy()
        missing_norm_data[missing_position_2d] = missvalue
        missing_norm_data = pd.DataFrame(missing_norm_data, columns=norm_data.columns) if isinstance(norm_data, pd.DataFrame) else missing_norm_data
        return missing_data, missing_norm_data

    return missing_data



def nan_filling(data):
    """
    用前方最近的观测值填补缺失值

    Args:
        data: (all_len, node_num)，numpy数组

    Returns:

    """
    df_data = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
    df_data = df_data.fillna(method="ffill", axis=0)
    df_data = df_data.fillna(method="backfill", axis=0)
    data = df_data.values if isinstance(data, np.ndarray) else df_data
    if np.isnan(data).any():
        raise ValueError("Data contains NaN values after filling.")

    return data