from pathlib import Path
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from minepy import MINE
from minepy import pstats, cstats
from copent import copent
import torch
import os
from utils.plot_heatmap import plot_adj_heatmap
from scipy.stats import kendalltau
from sklearn.feature_selection import mutual_info_regression


def A_w_calculate(args, data_normal):
    if isinstance(data_normal, np.ndarray):
        data = data_normal
    elif isinstance(data_normal, pd.DataFrame):
        data = data_normal.values
    elif isinstance(data_normal, torch.Tensor):
        data = data_normal.cpu().numpy()
    else:
        raise ValueError("data_normal should be np.ndarray or pd.DataFrame")
    ca_len, node_num = data.shape
    A_w = np.zeros((node_num, node_num)).astype(np.float32)
    if args.graph_ca_meth == "MIC":
        # for i in range(0, node_num - 1):
        #     for j in range(i + 1, node_num):
        #         mine = MINE(alpha=args.MIC_alpha, c=args.MIC_c, est="mic_approx")
        #         mine.compute_score(data[:, i], data[:, j])
        #         A_w[i, j] = mine.mic()
        #         A_w[j, i] = A_w[i, j]
        # np.fill_diagonal(A_w, 1)
        data_T = data.T
        mic_p, tic_p =  pstats(data_T, alpha=args.MIC_alpha, c=args.MIC_c, est="mic_e")
        A_w = np.zeros((node_num, node_num)).astype(np.float32)
        triu_idx = np.triu_indices(node_num, k=1)
        A_w[triu_idx] = mic_p
        A_w += A_w.T
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Copent":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                data1 = data[:, [i, j]]
                A_w[i, j] = copent(data1)
                A_w[j, i] = A_w[i, j]
        A_w = (A_w - A_w.min()) / (A_w.max() - A_w.min() + 1e-5)
        np.fill_diagonal(A_w, 1)
    elif args.graph_ca_meth == "Cosine":
        data_T = data.T
        A_w = np.matmul(data_T, data)
        A_w = A_w / ((np.linalg.norm(data_T, axis=1, keepdims=True) * np.linalg.norm(data, axis=0, keepdims=True)) + 1e-5)
        np.fill_diagonal(A_w, 1)
        A_w = np.abs(A_w)
    elif args.graph_ca_meth == "Kendall":
        for i in range(0, node_num - 1):
            for j in range(i + 1, node_num):
                tau, _ = kendalltau(data[:, i], data[:, j])
                if np.isnan(tau):
                    tau = 0
                A_w[i, j] = tau
                A_w[j, i] = A_w[i, j]
        np.fill_diagonal(A_w, 1)
        A_w = np.abs(A_w)
    elif args.graph_ca_meth == "MutualInfo":
        for i in range(0, node_num - 1):
            X = data[:, i+1:]
            y = data[:, i]
            mi = mutual_info_regression(X, y)
            A_w[i, i+1:] = mi
            A_w[i+1:, i] = mi
        A_w[A_w < 0] = 0
        A_w = A_w / A_w.max() if A_w.max() != 0 else A_w
        np.fill_diagonal(A_w, 1)
    else:
        raise ValueError("method should be MIC or Copent or Cosine")

    return A_w


def A_w_csv_and_plot(args, A_w, csv_dir=None):
    if not os.path.exists(csv_dir):
        Path(os.path.dirname(csv_dir)).mkdir(parents=True, exist_ok=True)
        A_w_df = pd.DataFrame(A_w)
        A_w_df.to_csv(csv_dir, index=False, header=False)

    file_path = csv_dir.replace('_A_w.csv', '_adj_heatmap.pdf')
    plot_adj_heatmap(A_w, file_path)

    return csv_dir


def A_other_calculate(args, A_w, if_return_norm=False):
    node_num = A_w.shape[0]
    A = np.zeros((node_num, node_num)).astype(np.float32)
    A[A_w >= args.graph_ca_thre] = 1
    np.fill_diagonal(A, 0)
    A_self = A + np.eye(node_num).astype(np.float32)

    for i in range(node_num):
        if np.sum(A[i]) == 0:
            max2_index = np.argsort(A_w[i])[-2]
            A[i, max2_index] = 1
            A[max2_index, i] = 1

    if if_return_norm:
        degree = np.sum(A, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        A_norm = np.matmul(np.matmul(D_inv_sqrt, A), D_inv_sqrt)
        degree_self = np.sum(A_self, axis=1)
        degree_self_inv_sqrt = np.power(degree_self, -0.5)
        degree_self_inv_sqrt[np.isinf(degree_self_inv_sqrt)] = 0
        D_self_inv_sqrt = np.diag(degree_self_inv_sqrt)
        A_self_norm = np.matmul(np.matmul(D_self_inv_sqrt, A_self), D_self_inv_sqrt)

        return A, A_self, A_w, A_norm, A_self_norm
    else:
        return A, A_self, A_w


def Graph_calculate(args, data_normal, if_return_norm=False, flag='train'):
    if args.graph_ca_meth == "MIC":
        file_name = args.graph_ca_meth + "_a" + str(args.MIC_alpha) + "_c" + str(args.MIC_c) \
                    + "_ca_len" + str(args.graph_ca_len)
    else:
        file_name = args.graph_ca_meth + "_ca_len" + str(args.graph_ca_len)
    if args.Decompose == 'STL':
        De_info = "_" + args.Decompose + "_season" + str(args.STL_seasonal)
        file_name = file_name + De_info
    else:
        De_info = "_" + args.Decompose + "_" + str(args.Wavelet_wave) + "_lv" + str(args.Wavelet_level)
        file_name = file_name + De_info
    if args.preMA:
        other_info = "_scale" + str(args.scale) + "_preMA_win" + str(args.preMA_win) \
                     + "_if_times" + str(args.if_timestamp)
    else:
        other_info = "_scale" + str(args.scale) + "_preMA" + str(args.preMA) \
                     + "_if_times" + str(args.if_timestamp)
    file_name = file_name + other_info
    csv_dir = args.table_save_path + '/' + file_name + "_A_w.csv"


    if flag == 'train':
        A_w = A_w_calculate(args, data_normal)
        A_w_csv_and_plot(args, A_w, csv_dir)
    elif flag in ['val', 'test']:
        "如果是val阶段，这里有个特殊情况，说明如下，但最终采用策略是：如果有A_w文件，则直接读取，否则计算A_w，但全程不保存"
        # 按理lighting会该先调用train_dataloader，然后再调用val_dataloader，
        # 但实际情况是lighting为了不需要等待漫长的训练过程才发现验证代码有错，https://zhuanlan.zhihu.com/p/120331610
        # 会在开始加载dataloader并开始训练时，就提前执行 “验证代码”：val_dataloader、validation_step、validation_epoch_end.
        # 这会导致此时还没有train_dataloader，还没有计算A_w，会报错，
        # 所以这里如果没有A_w文件，就先计算一个，但是不保存，只用来最初的走通验证代码
        if os.path.exists(csv_dir):
            A_w = pd.read_csv(csv_dir, header=None).values
        else:
            A_w = A_w_calculate(args, data_normal)
            print("there is no A_w file, so calculate A_w in {}".format(flag))

    if if_return_norm:
        A, A_self, A_w, A_norm, A_self_norm = A_other_calculate(args, A_w, if_return_norm)
    else:
        A, A_self, A_w = A_other_calculate(args, A_w, if_return_norm)

    if if_return_norm:
        return A, A_self, A_w, A_norm, A_self_norm
    else:
        return A, A_self, A_w
