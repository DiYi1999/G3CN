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
from utils.plot_sup import *

def MyPlot_AD(args, X_orig, Y_fore, label, all_label, S, detect_01, exam_result, args_dict):
    pass

def MyPlot_FC(args, X_orig, Y_fore, exam_result, args_dict):
    orig_tensor = X_orig.T.contiguous()
    fore_tensor = Y_fore.T.contiguous()

    plt.style.use(['science', 'ieee', 'high-vis'])

    plot_dirname = '_'.join([f'{k}={float(v):.3e}' for k, v in exam_result.items()])
    plot_dirname = plot_dirname[:66]
    plot_dirname_path = args.plot_save_path + '/' + plot_dirname + '.pdf'
    dirname = os.path.dirname(plot_dirname_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)

    pdf = PdfPages(plot_dirname_path)

    firstPage = plt.figure(figsize=(6, 6))
    firstPage.clf()
    txt = '@'.join([f'{k}={v}' for k, v in {**exam_result, **args_dict}.items()])
    hang = 18
    hang_len = len(txt) // hang
    txt = '\n'.join(txt[i * hang_len:(i + 1) * hang_len] for i in range(0, hang + 1))
    firstPage.text(0.5, 0.5, txt, ha='center', va='center')
    pdf.savefig(firstPage)
    plt.close()

    pdf = channels_plot_FC(orig_tensor=orig_tensor,
                           reco_tensor=fore_tensor,
                           data_name=args.data_name,
                           pdf=pdf)

    pdf.close()

def MyPlot_RE(args, X_orig, Y_fore, exam_result, args_dict, time_label=None):
    orig_tensor = X_orig.T.contiguous()
    fore_tensor = Y_fore.T.contiguous()

    plt.style.use(['science', 'ieee', 'high-vis'])

    plot_dirname = '_'.join([f'{k}={float(v):.3e}' for k, v in exam_result.items()])
    plot_dirname = plot_dirname[:66]
    plot_dirname_path = args.plot_save_path + '/' + plot_dirname + '.pdf'
    dirname = os.path.dirname(plot_dirname_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)

    pdf = PdfPages(plot_dirname_path)

    firstPage = plt.figure(figsize=(6, 6))
    firstPage.clf()
    txt = '@'.join([f'{k}={v}' for k, v in {**exam_result, **args_dict}.items()])
    hang = 18
    hang_len = len(txt) // hang
    txt = '\n'.join(txt[i * hang_len:(i + 1) * hang_len] for i in range(0, hang + 1))
    firstPage.text(0.5, 0.5, txt, ha='center', va='center')
    pdf.savefig(firstPage)
    plt.close()

    if time_label is not None:
        pdf = channels_plot_RE_T(orig_tensor=orig_tensor,
                                 reco_tensor=fore_tensor,
                                 data_name=args.data_name,
                                 time_label=time_label,
                                 pdf=pdf)
    else:
        pdf = channels_plot_RE(orig_tensor=orig_tensor,
                               reco_tensor=fore_tensor,
                               data_name=args.data_name,
                               pdf=pdf)

    pdf.close()
