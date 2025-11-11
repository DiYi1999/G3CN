import numpy as np
import seaborn as sns
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.font_manager._rebuild()
import scienceplots
# import latex
import os
from pathlib import Path
from matplotlib.colors import ListedColormap


def plot_adj_heatmap(A_w, save_path):
    """

        Args:
            A_w: (node_num, node_num)权重邻接矩阵，即相关性矩阵，可以做可视化用
            save_path: 保存路径, 以.pdf结尾
            file_name: 文件名

        Returns:

        """

    """相关设置"""
    plt.style.use(['science', 'ieee', 'high-vis'])
    heatmap_save_path = save_path
    dirname = os.path.dirname(heatmap_save_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)

    """预处理"""
    np.fill_diagonal(A_w, 1)
    A_w = np.floor(A_w * 100) / 100

    """绘图"""
    node_num = A_w.shape[0]
    if not os.path.exists(heatmap_save_path):
        # 展开pdf
        pdf = PdfPages(heatmap_save_path)

        for cmap in ['bwr', 'Purples', 'Blues']:
            for linecolor in ['black', 'white']:

                fig, ax = plt.subplots(figsize=(9/20*node_num, 6/20*node_num))
                sns.heatmap(data=A_w, annot=True, fmt=".2f", linewidths=.5, cmap=cmap, linecolor=linecolor, ax=ax)
                fig.patch.set_edgecolor('black')
                pdf.savefig(fig)
                plt.close()

                up_A_w = np.where(np.triu(np.ones(A_w.shape), k=0).astype(bool), A_w, np.nan)
                mask = np.zeros_like(A_w)
                mask[np.triu_indices_from(mask)] = True
                fig, ax = plt.subplots(figsize=(9/20*node_num, 6/20*node_num))
                sns.heatmap(data=A_w, mask=mask, annot=False,
                            linewidths=.5, cmap=cmap, linecolor=linecolor, ax=ax)
                sns.heatmap(data=up_A_w, cmap=ListedColormap(['white']), cbar=False,
                            annot=True, fmt=".2f", linewidths=.5, linecolor=linecolor, ax=ax)
                fig.patch.set_edgecolor('black')
                pdf.savefig(fig)
                plt.close()

        """关闭pdf"""
        pdf.close()