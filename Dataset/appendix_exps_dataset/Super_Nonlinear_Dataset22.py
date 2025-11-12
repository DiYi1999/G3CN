"""
Unlabeled
"""

import pandas as pd
import os
from pathlib import Path
import numpy as np



root_path = 'E:\OneDrive\DATA\\reconstruct'
super_nonlinear_dataset_path = 'E:\OneDrive\DATA\\reconstruct\Super_Nonlinear_Dataset22\Super_Nonlinear_Dataset22\Super_Nonlinear_Dataset22_train.csv'

typ_dataset_path = 'E:\OneDrive\DATA\\reconstruct\Typical_Nonlinear_Operators1\Typical_Nonlinear_Operators1\Typical_Nonlinear_Operators1_train.csv'
six_dataset_path = 'E:\OneDrive\DATA\\reconstruct\SixD_Hyperchaotic2\SixD_Hyperchaotic2\SixD_Hyperchaotic2_train.csv'
cart_dataset_path = 'E:\OneDrive\DATA\\reconstruct\Cart_Pendulum2\Cart_Pendulum2\Cart_Pendulum2_train.csv'

ETTh1_dataset_path = 'E:\OneDrive\DATA\\forecast\ETTh1\ETTh1\ETTh1.csv'
ETTh2_dataset_path = 'E:\OneDrive\DATA\\forecast\ETTh2\ETTh2\ETTh2.csv'
ETTm1_dataset_path = 'E:\OneDrive\DATA\\forecast\ETTm1\ETTm1\ETTm1.csv'
ETTm2_dataset_path = 'E:\OneDrive\DATA\\forecast\ETTm2\ETTm2\ETTm2.csv'


typ_data = pd.concat([pd.read_csv(typ_dataset_path.replace('train', split)) for split in ['train', 'val', 'test']], axis=0, ignore_index=True)
typ_data = typ_data.sort_values(by=typ_data.columns[0], ascending=True).reset_index(drop=True)
six_data = pd.concat([pd.read_csv(six_dataset_path.replace('train', split)) for split in ['train', 'val', 'test']], axis=0, ignore_index=True)
six_data = six_data.sort_values(by=six_data.columns[0], ascending=True).reset_index(drop=True)
cart_data = pd.concat([pd.read_csv(cart_dataset_path.replace('train', split)) for split in ['train', 'val', 'test']], axis=0, ignore_index=True)
cart_data = cart_data.sort_values(by=cart_data.columns[0], ascending=True).reset_index(drop=True)
ett_h1_data = pd.read_csv(ETTh1_dataset_path).drop(columns=['date'])
ett_h2_data = pd.read_csv(ETTh2_dataset_path).drop(columns=['date'])
ett_m1_data = pd.read_csv(ETTm1_dataset_path).drop(columns=['date'])
ett_m2_data = pd.read_csv(ETTm2_dataset_path).drop(columns=['date'])

min_length = min(len(typ_data), len(six_data), len(cart_data), len(ett_h1_data), len(ett_h2_data), len(ett_m1_data), len(ett_m2_data))
typ_data = typ_data.iloc[:min_length]
six_data = six_data.iloc[:min_length]
cart_data = cart_data.iloc[:min_length]
ett_h1_data = ett_h1_data.iloc[:min_length]
ett_h2_data = ett_h2_data.iloc[:min_length]
ett_m1_data = ett_m1_data.iloc[:min_length]
ett_m2_data = ett_m2_data.iloc[:min_length]

combined_data = pd.concat([typ_data, six_data, cart_data, ett_h1_data, ett_h2_data, ett_m1_data, ett_m2_data], axis=1)

# 打乱顺序
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

for train_or_test in ['train', 'val', 'test']:
    if train_or_test == 'train':
        split_data = combined_data.iloc[:int(0.6 * len(combined_data))]
    elif train_or_test == 'val':
        split_data = combined_data.iloc[int(0.6 * len(combined_data)):int(0.8 * len(combined_data))]
    else:  # test
        split_data = combined_data.iloc[int(0.8 * len(combined_data)):]
    # 保存数据
    save_path = super_nonlinear_dataset_path.replace('train', train_or_test)
    dirname = os.path.dirname(save_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    split_data.to_csv(save_path, index=False)
    print(f'拼接后的数据已保存至：{save_path}')

import sys
sys.path.append('E:/OneDrive - stu.xjtu.edu.cn/CODE/MyWorks_Codes/Data_analysis')
from MSL_SMAP.plot import channel_plot_x
for train_or_test in ['train', 'val', 'test']:
    save_path = super_nonlinear_dataset_path.replace('train', train_or_test)
    channel_plot_x(save_path.replace('.csv', '.pdf'), pd.read_csv(save_path))
    print(f'图像已保存至：{save_path.replace(".csv", ".pdf")}')

# 结束
print('拼接数据集完成')
















