"""
CP是Cart Pendulum，位置x, 角度theta, 速度v, 角速度omega, 加速度a, 角加速度alpha
9 channels: t, x_in, theta_in, v_in, omega_in, v_out, omega_out, a_out, alpha_out
the time order has been shuffled
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
from scipy.integrate import solve_ivp



def CartPendulum(t, X):
    x, theta, v, omega = X
    # position x, angle theta and vel. v and angle vel. omega
    m=1  # mass of pendulum 摆的质量
    M=1  # mass of cart 车的质量
    R=1  # length of rod (constant) 杆长
    g=9.81  # gravity 加速度
    d=0  # cart friction constant 车摩擦
    b=0   # pendulum friction constant 摆阻尼
    k=6  # spring stiffness 弹簧刚度
    sin=np.sin
    cos=np.cos
    sqr=np.square
    '计算输出'
    out1 = v
    out2 = omega
    out3 = (
                      m*R*sqr(omega)*sin(theta)
                      + m*g*sin(theta)*cos(theta)
                      - k*x - d*v + b/R*omega*cos(theta)
              ) \
              / (M + m*sqr(sin(theta)))
    out4 = (
                      -m*R*sqr(omega)*sin(theta)*cos(theta)
                      - (m+M)*g*sin(theta) + k*x*cos(theta)
                      + d*v*cos(theta) - (1 + M/m)*b/R*omega
              ) \
                /(R*(M + m*sqr(sin(theta))))

    return [out1, out2, out3, out4]



"""生成数据"""
data = pd.DataFrame()
tspan = [0, 40]
# 仿真40s
num = 20000
h_step = (tspan[1] - tspan[0]) / num
t = np.arange(tspan[0], tspan[1] + h_step, h_step)

X_init = np.array([1, 0, 0, 0])
solve = solve_ivp(CartPendulum, tspan, X_init, method="RK45", t_eval=t)
IN = solve.y.T
OUT = np.array([CartPendulum(0, IN[i,:]) for i in range(IN.shape[0])])

data = pd.DataFrame(np.concatenate([t.reshape(-1,1), IN, OUT], axis=1)
                    , columns=['t', 'x_in', 'theta_in', 'v_in', 'omega_in', 'v_out', 'omega_out', 'a_out', 'alpha_out'])
# data: pandas.DataFrame: (20000,9)

"""打乱数据"""
data = data.sample(frac=1).reset_index(drop=True)
# .sample(frac=1)是打乱顺序，reset_index(drop=True)是重置索引

"""归一化/标准化  不做了，要求使用者在dataset中自己预处理，自己选用标准化/归一化"""

"""分割数据，比例7：1：2"""
train = data.iloc[:14000]
'(14000,9)'
valid = data.iloc[14000:16000]
'(2000,9)'
test = data.iloc[16000:]
'(4000,9)'

"""保存数据"""
"""先设置路径"""
save_path = 'E:\DY-YDdrive\DATA' \
            + '\\reconstruct\Cart_Pendulum2\Cart_Pendulum2' \
            + '\Cart_Pendulum2_train.csv'
dirname = os.path.dirname(save_path)
Path(dirname).mkdir(parents=True, exist_ok=True)
# 保存train
train.to_csv(save_path, index=False)
# 保存valid
val_path = save_path.replace('train', 'val')
valid.to_csv(val_path, index=False)
# 保存test
test_path = save_path.replace('train', 'test')
test.to_csv(test_path, index=False)
print('数据已保存至：', save_path)

"""画图"""
# 画train
channel_plot_x(save_path.replace('.csv', '.pdf'), train)
# 画valid
channel_plot_x(val_path.replace('.csv', '.pdf'), valid)
# 画test
channel_plot_x(test_path.replace('.csv', '.pdf'), test)
print('图像已保存至：', save_path.replace('.csv', '.pdf'))




