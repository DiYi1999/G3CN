"""
Typical_Nonlinear_Operators1
14 channels: t, x, linear, addition, multiplication, parabolic, cubic, polynomial, exponential, logarithmic, sinusoidal, VF sinusoidal, hybrid, random
the time order has been shuffled,
No normalization/standardization has been done, so user should preprocess the data
"""



import csv
import numpy as np
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from MSL_SMAP.plot import *



"""先设置路径"""
train_dir_path = 'E:\DY-YDdrive\DATA' + '\\reconstruct\Typical_Nonlinear_Operators1\Typical_Nonlinear_Operators1' + '\Typical_Nonlinear_Operators1_train.csv'
dirname = os.path.dirname(train_dir_path)
Path(dirname).mkdir(parents=True, exist_ok=True)

"""生成数据"""
num = 20000
data = pd.DataFrame()
rs = np.random.RandomState(seed=42)

t = np.linspace(0, 1, num, endpoint=False)
data['t'] = t


x = t
data['x'] = x
# linear
data['linear'] = 2*x + rs.uniform(-0.02, 0.02, num)
# addition
data['addition'] = data['x'] + data['linear']
# multiplication
data['multiplication'] = data['x'] * data['linear']
# parabolic
data['parabolic'] = 4*(x-0.5)**2
# cubic
data['cubic'] = 8*(x-0.5)**3
# polynomial
data['polynomial'] = 0.5*(3*x-2)**3 + 1*(3*x-2)**2 - 0.5*(3*x-2) - 0.5
# exponential
data['exponential'] = 0.3 * np.exp(x)
# logarithmic
data['logarithmic'] = np.log(x+0.1)
# sinusoidal
data['sinusoidal'] = np.cos(2 * np.pi * x)
# data['sinusoidal'] = np.cos(4 * np.pi * x)
# VF sinusoidal
data['VF sinusoidal'] = np.sin(3 * np.pi * x * (x-1))
# hybrid
data['hybrid'] = 0.5 * np.sin(2 * np.pi * x) + x
# random
rs = np.random.RandomState(seed=66)
data['random'] = rs.uniform(0, 1, num)

"""打乱顺序"""
data = data.sample(frac=1).reset_index(drop=True)
# .sample(frac=1)是打乱顺序，reset_index(drop=True)是重置索引

"""分割数据，比例7：1：2"""
train_data = data.iloc[:14000]
valid_data = data.iloc[14000:16000]
test_data = data.iloc[16000:]

"""保存数据"""
train_data.to_csv(train_dir_path, index=False)
val_dir_path = train_dir_path.replace('train', 'val')
valid_data.to_csv(val_dir_path, index=False)
test_dir_path = train_dir_path.replace('train', 'test')
test_data.to_csv(test_dir_path, index=False)
print('数据已保存至：', train_dir_path)

"""画图"""
channel_plot_x(train_dir_path.replace('.csv', '.pdf'), train_data)
channel_plot_x(val_dir_path.replace('.csv', '.pdf'), valid_data)
channel_plot_x(test_dir_path.replace('.csv', '.pdf'), test_data)
print('图像已保存至：', train_dir_path.replace('.csv', '.pdf'))









