"""
Labeled


A01_label = np.zeros((120, 120)).astype(np.float32)
for i in range(0, 120, 2):
    A01_label[i, i] = 1.0
    A01_label[i+1, i+1] = 1.0
    A01_label[i, i+1] = 1.0
    A01_label[i+1, i] = 1.0
"""



import pandas as pd
import os
from pathlib import Path
import numpy as np
import os
from pathlib import Path
import numpy as np
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
import scienceplots



root_path = 'E:\OneDrive\DATA\\reconstruct'
super_nonlinear_dataset_path = 'E:\OneDrive\DATA\\reconstruct\Super_Nonlinear_Dataset33\Super_Nonlinear_Dataset33\Super_Nonlinear_Dataset33_train.csv'

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

random_seed = 42
combined_data = pd.DataFrame()
for i in range(1, typ_data.shape[1]-1):
    two_rows = pd.concat([typ_data.iloc[:, 0:1], typ_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)
for i in range(1, six_data.shape[1]):
    two_rows = pd.concat([six_data.iloc[:, 0:1], six_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)
for i in range(1, cart_data.shape[1]):
    two_rows = pd.concat([cart_data.iloc[:, 0:1], cart_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)

t_df = pd.DataFrame(np.linspace(0, 1, min_length), columns=['timestamp'])

for i in range(ett_h1_data.shape[1]):
    two_rows = pd.concat([t_df, ett_h1_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)
for i in range(ett_h2_data.shape[1]):
    two_rows = pd.concat([t_df, ett_h2_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)
for i in range(ett_m1_data.shape[1]):
    two_rows = pd.concat([t_df, ett_m1_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)
for i in range(ett_m2_data.shape[1]):
    two_rows = pd.concat([t_df, ett_m2_data.iloc[:, i:i+1]], axis=1)
    two_rows = two_rows.sample(frac=1, random_state=random_seed+int(combined_data.shape[1])).reset_index(drop=True)
    combined_data = pd.concat([combined_data, two_rows], axis=1)

# combined_data = pd.concat([typ_data, six_data, cart_data, ett_h1_data, ett_h2_data, ett_m1_data, ett_m2_data], axis=1)
# # 打乱顺序
# combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# （len，120）维度的dataframe，以两列为一组，进行第二维的顺序打乱
new_order = np.random.RandomState(seed=666).permutation(combined_data.shape[1] // 2)
shuffled_data = pd.DataFrame()
for idx in new_order:
    shuffled_data = pd.concat([shuffled_data, combined_data.iloc[:, idx*2:idx*2+2]], axis=1)
combined_data = shuffled_data


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
for train_or_test in ['train', 'val', 'test']:
    save_path = super_nonlinear_dataset_path.replace('train', train_or_test)
    data = pd.read_csv(save_path)
    channel_list = data.columns.tolist()
    plot_dirname_path = save_path.replace('.csv', '.pdf')

    plt.style.use(['science', 'ieee', 'high-vis'])
    if not plot_dirname_path.endswith('.pdf'):
        raise ValueError('plot_dirname_path should end with .pdf')
    pdf = PdfPages(plot_dirname_path)
    # plt.rcParams['figure.figsize'] = data.shape[0] * 6/15000, 2
    plt.rcParams['figure.figsize'] = 6, 1.5

    for dim in range(data.shape[1] // 2):
        x = data.values[:, dim * 2]
        y = data.values[:, dim * 2 + 1]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=0.1, color='k')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        if channel_list is not None:
            ax.set_title(channel_list[dim * 2 + 1])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()
    pdf.close()

    print(f'图像已保存至：{save_path.replace(".csv", ".pdf")}')

# 结束
print('拼接数据集完成')


















