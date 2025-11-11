"""
6D_Hyperchaotic
13 channel： t, x, y, z, u, v, w, x_dot, y_dot, z_dot, u_dot, v_dot, w_dot
the time order has been shuffled
no normalization/standardization has been done, so user should preprocess the data
"""



import csv
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from MSL_SMAP.plot import *
from scipy.integrate import solve_ivp


"""# Function for the Lorenz system"""
def f(t, y):
    x, y, z, u, v, w = y
    return np.array([
        a * (y - x) + u,
        c * x - x * z - y - v,
        x * y - b * z,
        d * u - y * z,
        r * y,
        -e * w + z * u
    ])


"""# Parameters"""
a = 10
b = 2.6667
c = 28
d = -1
e = 10
r = 3

"""# Define the time span and step size"""
num = 20000
tspan = [0, 10]
h_step = (tspan[1] - tspan[0]) / num
t = np.arange(tspan[0], tspan[1] + h_step, h_step)

"""# Initialize arrays for the solution"""
x, y, z, u, v, w = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

"""# Set initial values"""
x[0], y[0], z[0], u[0], v[0], w[0] = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

"""# Implementing Runge-Kutta method"""
for i in range(len(t) - 1):
    k1 = h_step * f(t[i],
                    [x[i], y[i], z[i], u[i], v[i], w[i]])
    k2 = h_step * f(t[i] + h_step / 2,
                    [x[i] + k1[0] / 2, y[i] + k1[1] / 2, z[i] + k1[2] / 2, u[i] + k1[3] / 2, v[i] + k1[4] / 2, w[i] + k1[5] / 2])
    k3 = h_step * f(t[i] + h_step / 2,
                    [x[i] + k2[0] / 2, y[i] + k2[1] / 2, z[i] + k2[2] / 2, u[i] + k2[3] / 2, v[i] + k2[4] / 2, w[i] + k2[5] / 2])
    k4 = h_step * f(t[i] + h_step,
                    [x[i] + k3[0], y[i] + k3[1], z[i] + k3[2], u[i] + k3[3], v[i] + k3[4], w[i] + k3[5]])

    x[i + 1] = x[i] + (1 / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    y[i + 1] = y[i] + (1 / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    z[i + 1] = z[i] + (1 / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    u[i + 1] = u[i] + (1 / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    v[i + 1] = v[i] + (1 / 6) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
    w[i + 1] = w[i] + (1 / 6) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

"""# Analytical right-hand sides (assuming these are meant to be the same as the numerical solution for comparison)"""
rhs_1_anal = a * (y - x) + u
rhs_2_anal = c * x - x * z - y - v
rhs_3_anal = x * y - b * z
rhs_4_anal = d * u - y * z
rhs_5_anal = r * y
rhs_6_anal = -e * w + z * u

"""# pandas DataFrame"""
df_data = pd.DataFrame({
    't': t,
    'x': x,
    'y': y,
    'z': z,
    'u': u,
    'v': v,
    'w': w,
    'x_dot': rhs_1_anal,
    'y_dot': rhs_2_anal,
    'z_dot': rhs_3_anal,
    'u_dot': rhs_4_anal,
    'v_dot': rhs_5_anal,
    'w_dot': rhs_6_anal,
    # 't': t,
    # 'h_step': h_step
})


df_data = df_data.sample(frac=1).reset_index(drop=True)
# .sample(frac=1)是打乱顺序，reset_index(drop=True)是重置索引

"""****归一化/标准化  不做了，要求使用者在dataset中自己预处理，自己选用标准化/归一化****"""

"""split"""
train_data = df_data.iloc[:14000]
# '(14000,13)'
valid_data = df_data.iloc[14000:16000]
# '(2000,13)'
test_data = df_data.iloc[16000:]
# '(4000,13)'

"""先设置路径"""
save_path = 'E:\DY-YDdrive\DATA' \
            + '\\reconstruct\SixD_Hyperchaotic2\SixD_Hyperchaotic2' \
            + '\SixD_Hyperchaotic2_train.csv'
dirname = os.path.dirname(save_path)
Path(dirname).mkdir(parents=True, exist_ok=True)

"""保存数据"""
train_data.to_csv(save_path, index=False)
val_path = save_path.replace('train', 'val')
valid_data.to_csv(val_path, index=False)
test_path = save_path.replace('train', 'test')
test_data.to_csv(test_path, index=False)
print('数据已保存至：', save_path)

"""画图"""
# 画train
channel_plot_x(save_path.replace('.csv', '.pdf'), train_data)
# 画valid
channel_plot_x(val_path.replace('.csv', '.pdf'), valid_data)
# 画test
channel_plot_x(test_path.replace('.csv', '.pdf'), test_data)
print('画图完成')















