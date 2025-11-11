import os
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import matplotlib.font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scienceplots
from scipy.interpolate import make_interp_spline


def channels_plot_AD(orig_tensor, reco_tensor, anomaly_label_tensor, anomaly_detect_tensor, anomaly_score_tensor,
                    data_name=None, name_list=None, pdf=None):
    pass


def channels_plot_FC(orig_tensor, reco_tensor,
                    data_name=None, name_list=None, pdf=None):
    plt.rcParams['figure.figsize'] = 6, 1.5
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_title(name_list[dim] + ' channel of ' + data_name)
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.plot(y_t, linewidth=1.0, label='Ground Truth', color='k')
        ax1.plot(y_p, '-', alpha=0.6, linewidth=1.0, label='Prediction', color='r')
        ax1.legend(ncol=2, bbox_to_anchor=(1.05, 1.1), loc="lower right")
        pdf.savefig(fig)
        plt.close()

    return pdf


def channels_plot_RE(orig_tensor, reco_tensor,
                     data_name=None, name_list=None, pdf=None):
    plt.rcParams['figure.figsize'] = 6, 1.5
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_title(name_list[dim] + ' channel of ' + data_name)
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.plot(y_t, linewidth=1.0, label='Ground Truth', color='k')
        ax1.plot(y_p, '-', alpha=0.6, linewidth=1.0, label='Prediction', color='r')
        ax1.legend(ncol=2, bbox_to_anchor=(1.05, 1.1), loc="lower right")
        pdf.savefig(fig)
        plt.close()

    return pdf


def channels_plot_RE_T(orig_tensor, reco_tensor,
                       data_name=None, name_list=None, time_label=None,
                       pdf=None):
    plt.rcParams['figure.figsize'] = 4, 2

    time_label = time_label.cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(time_label, linewidth=1.0, color='k')
    ax.set_xlabel('Step')
    ax.set_ylabel('Timestamp')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    for dim in range(orig_tensor.shape[1]):
        y_t, y_p = orig_tensor[:, dim].cpu().numpy(), \
                   reco_tensor[:, dim].cpu().numpy()
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_xlabel('Timestamp')
        if name_list is not None:
            ax1.set_title(name_list[dim] + ' channel of ' + data_name)
        else:
            ax1.set_title('channel ' + str(dim+1))
        ax1.scatter(time_label, y_t, s=0.1, label='Ground Truth', color='k')
        ax1.scatter(time_label, y_p, s=0.1, label='Prediction', color='r')
        ax1.legend(ncol=2, bbox_to_anchor=(1.05, 1.2), loc="lower right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

    return pdf


def channels_plot_onepage(orig_sample_tensor, ratio=None, color='k', pdf=None):
    if ratio is None:
        ratio = 1
    plt.rcParams['figure.figsize'] = 3, 1*orig_sample_tensor.shape[1]
    fig, axs = plt.subplots(orig_sample_tensor.shape[1], 1, sharex=True)
    for dim, ax in enumerate(axs):
        ax.plot(orig_sample_tensor[:, dim].cpu().numpy(), linewidth=1.0, color=color)
        ax.set_ylabel('channel' + str(dim))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    return pdf


def score_kernel_density_plot(anomaly_score_tensor, anomaly_label_vector, pdf):
    pass


def Plot_x_y_scatter(dataframe, x_name, y_name, xlabel, ylabel, title, fig_size, pdf, legend_ncols=1, legend_outside=False, xlim=None, ylim=None):
    plt.style.use(['science', 'ieee', 'high-vis'])
    x = dataframe[x_name].values
    markers = ['o', 's', '^', 'D', 'v', 'x', '*', 'p', 'h', '+']
    if y_name is not None and isinstance(y_name, str):
        y_name = [y_name]
    
    colors = sns.color_palette("hls", len(y_name)).as_hex()  
    if len(y_name) < len(markers):
        markers = markers[:len(y_name)]
    elif len(y_name) > len(markers):
        markers = markers * (len(y_name) // len(markers) + 1) 
        markers = markers[:len(y_name)]
    else:
        markers = markers

    fig, ax = plt.subplots(figsize=fig_size)

    if len(y_name) == 2:
        ax.plot(x, dataframe[y_name[0]].values,
                marker=markers[0], color=colors[0], markersize=10, linewidth=3,
                markerfacecolor=colors[0], markeredgecolor='black', markeredgewidth=1.5,
                label=y_name[0] + ' (left y-axis)')
        ax2 = ax.twinx()
        ax2.plot(x, dataframe[y_name[1]].values,
                 marker=markers[1], color=colors[1], markersize=10, linewidth=3,
                 markerfacecolor=colors[1], markeredgecolor='black', markeredgewidth=1.5,
                 label=y_name[1] + ' (right y-axis)')
    else:
        for y_legend, color, marker in zip(y_name, colors, markers):
            ax.plot(x, dataframe[y_legend].values, 
                    marker=marker, color=color, markersize=10, linewidth=3,
                    markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5, 
                    label=y_legend)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title is not None: ax.set_title(title)
    
    if len(y_name) == 2:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False)
    elif len(y_name) == 1:
        ax.legend(frameon=False)
    else:
        if legend_outside:
            ax.legend(frameon=False, 
                ncols=legend_ncols,
                bbox_to_anchor=(0.5, 1.01),
                loc='lower center')
        else:
            ax.legend(frameon=False, ncols=legend_ncols)
    
    ax.grid(True, linestyle='--', alpha=0.8)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    return pdf
