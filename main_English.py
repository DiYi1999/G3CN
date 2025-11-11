import os
from argparse import ArgumentParser
import math
from data.MyDataset import *
from model.mycallback import MyEarlyStopping
from model.MyModel import *
import torch
"Don't touch the following codes"
def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from data.lightingdata import MyLigDataModule
import numpy as np
from ray import air, tune
from ray.tune.search.ax import AxSearch
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
### This TuneReportCheckpointCallback is not updated by official, still using pytorch_lightning, while my lighting2.0 uses pytorch.lightning
import ptwt, pywt
from main_sub import *


def set_args():
    parser = ArgumentParser()

    ### Task Settings
    parser.add_argument('--TASK', type=str, default='forecast',
                        help='anomaly_detection or forecast or reconstruct'
                             'The relationship between TASK and BaseOn below is:'
                             'If TASK is anomaly_detection, anomaly detection can be based on reconstruct or forecast, label data file needed'
                             'If TASK is forecast, then BaseOn can only be forecast, no label data file needed'
                             'If TASK is reconstruct, then BaseOn can only be reconstruct, no label data file needed')
    parser.add_argument('--BaseOn', type=str, default='forecast',
                        help='reconstruct or forecast')
    parser.add_argument('--data_name', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--Decompose', type=str, default='None', help='None/STL/WaveletPacket/Wavelet')
    # if TASK == 'forecast':
    parser.add_argument('--channel_to_channel', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'MS:multivariate predict univariate, S:univariate predict univariate')
    parser.add_argument('--MS_which', type=int, default=1, help='if MS is selected above, need to specify which single column to use')
    # if TASK == 'reconstruct':
    parser.add_argument('--reco_form', type=str, default='all_to_all', help='options:[all_to_all, half_to_half]')
    parser.add_argument('--if_time_dimOne', type=bool, default=True,
                        help='whether dataset time is disordered, if True, default first dimension of dataset is time label, this dimension will not participate in reconstruction, but used for plotting as x-axis, otherwise the plot would be messy')



    ### Experiment Settings
    parser.add_argument('--Version', type=str, default='V1.0', help='code version V1.0')
    parser.add_argument('--Method', type=str, default='MadjGCN_Project', help='project name, top-level directory')
    exp_name = parser.parse_known_args()[0].Version + '_' + \
               parser.parse_known_args()[0].Method + '_' + \
               parser.parse_known_args()[0].data_name + '_' + \
               parser.parse_known_args()[0].Decompose + '_' + \
               parser.parse_known_args()[0].TASK
    parser.add_argument('--exp_name', type=str, default=exp_name, help='experiment name/excel filename/save folder name etc')



    ### Data Import Path
    """f = open(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_path)), "rb")"""

    parser.add_argument('--root_path', type=str, default='/home/data/DiYi/DATA',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=parser.parse_known_args()[0].TASK + '/' +
                                                         parser.parse_known_args()[0].data_name + '/' +
                                                         parser.parse_known_args()[0].data_name,
                        help='ETTh1/ETTh1')



    ### Path Related
    parser.add_argument('--result_root_path', type=str, default='/home/data/DiYi/MyWorks_Results',
                        help='save path for experiment results')
    save_path = parser.parse_known_args()[0].result_root_path + '/' + \
                parser.parse_known_args()[0].Method + '/' + \
                parser.parse_known_args()[0].Version + '/' + \
                parser.parse_known_args()[0].data_name + '/' + \
                parser.parse_known_args()[0].exp_name
    parser.add_argument('--ckpt_save_path', type=str, default= save_path + '/ckpt',
                        help='checkpoint cache path, also log save path')
    parser.add_argument('--table_save_path', type=str, default=save_path + '/table',
                        help='save path for debugging csv/excel files')
    parser.add_argument('--plot_save_path', type=str, default=save_path + '/plot',
                        help='save path for plots')



    ### node_num Related Strategy Formulation
    # If Decompose above is not None, STL/Wavelet decomposition parameters
    parser.add_argument('--STL_seasonal', type=int, default=7,
                        help='Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).')
    parser.add_argument('--Wavelet_wave', type=str, default='db4', help='wavelet basis function, default db4')
    parser.add_argument('--Wavelet_level', type=int, default=2, help='wavelet decomposition levels')
    parser.add_argument('--if_timestamp', type=bool, default=True, help='whether to use timestamp data during forecast or reconstruct')

    sensor_num, node_num, timestamp_dim = set_node_num(parser.parse_known_args()[0].data_name,
                                                       parser.parse_known_args()[0].Decompose,
                                                       parser.parse_known_args()[0].Wavelet_level,
                                                       parser.parse_known_args()[0].if_timestamp,
                                                       parser.parse_known_args()[0].reco_form)



    ### Data
    parser.add_argument('--Dataset', default= parser.parse_known_args()[0].data_name + '_Dataset',
                        help='which Dataset to use from MyDataset.py file')
    parser.add_argument('--dataset_split_ratio', type=int, default=0.8, help='dataset split ratio, train and validation set proportion')

    parser.add_argument('--sensor_num', type=int, default=sensor_num, help='sensor_num')
    parser.add_argument('--node_num', type=int, default=node_num, help='node_num')
    parser.add_argument('--timestamp_dim', type=int, default=timestamp_dim,
                        help='dataset will additionally return timestamp, storing year, month, day, hour, minute, second etc')
    # Sliding window and sliding window step
    parser.add_argument('--lag', type=int, default=24, help='lag, i.e. seq_len/slide_win sliding window length')  #
    lag = parser.parse_known_args()[0].lag
    parser.add_argument('--lag_step', type=int, default=1, help='sliding window step when sampling from those datasets, default 1')  #
    # if TASK == 'forecast':
    parser.add_argument('--label_len', type=int, default=1, help='if forecast task, Transformer models need hint segment, other models do not')  #
    parser.add_argument('--pred_len', type=int, default=24, help='if forecast task, predict how many steps')
    parser.add_argument('--pred_step', type=int, default=96,
                        help='this parameter only works when forecast length pred_len is greater than input time step length lag, because multiple RNN loops are needed to generate sufficiently long pred_len'
                             'each loop predicts pred_step time steps, specifically, pred_len divided by pred_step rounded up is the number of loops, default value consistent with lag, or half of lag')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--features', type=str, default='M', help='[S, M], former uses only partial sensor data, M uses all')
    parser.add_argument('--target', type=str, default='OT', help='if S, select which sensor/feature to use')
    # parser.add_argument('--missing_rate', type=float, default=0, help='data missing rate')  #
    # parser.add_argument('--missvalue', default=np.nan, help='when artificially creating missing data, fill missing positions with np.nan or 0')



    ### Preprocessing
    parser.add_argument('--scale', type=bool, default=True, help='whether to standardize/normalize data')
    parser.add_argument('--inverse', type=bool, default=False, help='whether to inverse standardize for recovery after test completion')
    # parser.add_argument('--preIDW', type=bool, default=True, help='whether to perform IDW interpolation preprocessing on missing data')
    parser.add_argument('--preMA', type=bool, default=False, help='whether to perform moving average preprocessing on noisy data')
    parser.add_argument('--preMA_win', type=int, default=5, help='moving average window size')



    ### Graph Structure Construction
    parser.add_argument('--graph_ca_len', type=int, default=1000,
                        help='when building graph, read how long training data segment for calculation, set this larger, because MIC needs large amount of data, 3000 is good')
    parser.add_argument('--graph_ca_meth', type=str, default='MIC',
                        help='MIC: Maximum Information Coefficient / Copent: Copent entropy / Cosine: Cosine similarity')
    parser.add_argument('--graph_ca_thre', type=float, default=0.6,
                        help='when building graph, MIC/Copula threshold selection basis, corrrlation weight above threshold are retained')
    parser.add_argument('--self_edge', type=bool, default= False,
                        help='whether to use self-connected graph, this function overlaps somewhat with residual function below, use selectively')
    parser.add_argument('--graph_if_norm_A', type=bool, default=False,
                        help='when building graph, whether to normalize adjacency matrix, default False, H2GCN must be False, but GPRGNN, GCN must be True')
    # MIC parameters
    parser.add_argument('--MIC_alpha', type=float, default=0.6, help='MIC parameter alpha')
    parser.add_argument('--MIC_c', type=int, default=15, help='MIC parameter c')



    ### Shared Parameters for All Methods
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--LeakyReLU_slope', type=float, default=1e-3, help='negative slope of LeakyReLU activation function, default 1e-2, if 0 then ReLU')



    ### If args.Method selects OursMethod:
    parser.add_argument('--Architecture', type=str, default='Series_TS', help='Parallel/Series_ST/Series_TS/Series_STS')

    ## if Parallel
    parser.add_argument('--fusion_method', type=str, default='MLP_Concat', help='MLP_Concat/Gate_Weight/Add_Minus')
    parser.add_argument('--fusion_hidden_dim', type=int, default=node_num * 6, help='hidden layer dimension of MLP_Concat')

    ## Spatial_block
    parser.add_argument('--spatial_method', type=str, default='MAdjGCN',
                        help='MAdjGCN/CMTS_GCN/MAdjGCN_Lite/GCN_s/Muti_S_GAT/GPRGNN/H2GCN/None' \
                        '【【【MAdjGCN is G3CN, CMTS_GCN is multi-layer G3CN】】】')
    parser.add_argument('--block_residual', type=float, default=0,
                        help='coefficient for residual connection before and after spatial convolution block, default 0/1')
    # if MAdjGCN or MAdjGCN_Lite, MAdjGCN is G3CN
    parser.add_argument('--K', type=int, default=node_num * 4, help='number of AdjGCN, according to UAT, must be large enough! try 32, 64, 128, 256')
    parser.add_argument('--residual_alpha', type=float, default=0,
                        help='coefficient for residual connection inside spatial convolution block, before entering activation function, default 0')
    # if CMTS_GCN, CMTS_GCN is multi-layer G3CN
    parser.add_argument('--CMTS_GCN_K_nums', type=list, default=[node_num*3, node_num*3],
                        help='CMTS_GCN is multi-layer MAdjGCN, this is list of K values (hidden neuron numbers) for each layer, ' \
                        'deepening network depth helps reduce network width burden, try [64,64], [128,128], [256,256], even more layers [256,256,256] etc')
    parser.add_argument('--CMTS_GCN_residual', type=float, default=0,
                        help='coefficient for residual connection between CMTS_GCN layers, default 0')
    # if GCN_s
    parser.add_argument('--GCN_layer_nums', type=list, default=[lag, lag],
                        help='e.g. [64, 64], means two GCN layers, each with 64 hidden neurons, ' \
                             'note, if forecast task, GCN_layer_nums[-1] cannot be less than pred_len')
    # if Muti_S_GAT
    parser.add_argument('--S_GAT_K', type=int, default=3, help='number of GAT heads, default 1')
    parser.add_argument('--S_GAT_embed_dim', type=int, default=96, help='embed_dim for GAT weight calculation')
    # if GIN
    parser.add_argument('--GIN_layer_nums', type=list, default=[lag*2, lag*2], help='e.g. [32,16,8]')
    parser.add_argument('--GIN_MLP_layer_num', type=int, default=1, help='number of MLP layers in each GIN layer formula')
    # if SGC
    parser.add_argument('--SGC_hidden_dim', type=int, default=int(lag*0.5), help='SGC hidden layer dimension')
    parser.add_argument('--SGC_K', type=int, default=3, help='SGC K value')
    # if GPRGNN
    parser.add_argument('--GPRGNN_K', type=int, default=3, help='GPRGNN K value, default 3')
    # if H2GCN
    parser.add_argument('--H2GCN_embed_dim', type=int, default=int(lag * 2), help='H2GCN encoding feature length')
    parser.add_argument('--H2GCN_round_K', type=int, default=3, help='H2GCN concatenation round')

    ## Temporal_block
    parser.add_argument('--temporal_method', type=str, default='GRU', help='TCN/GRU/Muti_T_GAT/None')
    # if TCN
    parser.add_argument('--TCN_layers_channels', type=list, default=[node_num, node_num],
                        help='e.g. [24, 8, 1], represents 3 layers, channel numbers change from 27 to 24 to 8 to 1.')
    parser.add_argument('--TCN_kernel_size', type=int, default=2, help='TCN convolution kernel size, default 2')
    # if GRU
    parser.add_argument('--GRU_layers', type=int, default=1, help='how many GRU')
    parser.add_argument('--GRU_hidden_num', type=int, default=64, help='GRU h dimension, default equals number of sensors')
    # if Muti_T_GAT
    parser.add_argument('--use_gatv2', type=bool, default=True, help='use GATV2 or GAT, GATV2 is improved version of GAT')
    parser.add_argument('--T_GAT_K', type=int, default=3, help='number of GAT heads, default 1')
    parser.add_argument('--T_GAT_embed_dim', type=int, default=parser.parse_known_args()[0].lag,
                        help='embed_dim for GAT weight calculation')



    ### Anomaly Score Calculation
    parser.add_argument('--how_precision', type=bool, default=False, help='Deprecated parameter')
    parser.add_argument('--focus_on', type=str, default='F1', help='Deprecated parameter')
    parser.add_argument('--S_moving_average_window', type=int, default=3, help='Deprecated parameter')




    ### Optimizer
    parser.add_argument('--optimizer', default=torch.optim.Adam, help='optimizer used, default torch.optim.Adam')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str,
                        help='which learning rate decay strategy to use, ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts')



    ### Training Configuration
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--patience', type=int, default=5, help='stop training if loss does not decrease for 10 consecutive epochs')
    parser.add_argument('--max_epoch', type=int, default=500, help='max epoch')



    ### Ray Tune
    # Hyperparameter search plan ASHAScheduler, Trial is one attempt
    # parser.add_argument('--max_trail', type=int, default=500, help='maximum number of trails')
    parser.add_argument('--trail_grace_period', type=int, default=40,
                        help='minimum number of trials to run before starting to reduce trial count')
    # parser.add_argument('--trail_time_out', type=int, default=600, help='maximum time (seconds) for trial to run')
    parser.add_argument('--trail_reduction_factor', type=int, default=3,
                        help='reduction_factor represents the proportion of trial reduction each time, '
                             'see https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/')
    parser.add_argument('--grid_num_samples', type=int, default=100, help='if -1, will repeat search infinitely until stop condition is met')


    # parser.set_defaults(max_epochs=100)
    args = parser.parse_args()

    return args




"""
Define main function, standalone version
"""
def main(devices):
    args = set_args()
    print("Please confirm if sensor number is:", args.sensor_num, "Please confirm if node number i.e. channel number is:", args.node_num)

    datamodule = MyLigDataModule(args)
    # model = torch.compile(MyLigModel(args))
    model = MyLigModel(args)

    trainer = pl.Trainer(strategy="auto",
                         # accelerator="gpu",
                         # devices="auto",
                         # devices=[0, 1, 2],
                         # devices=[0, 3],
                         devices=devices,
                         fast_dev_run=False,
                         max_epochs=args.max_epoch,
                         # callbacks=[pl.callbacks.EarlyStopping(monitor="training_loss", patience=args.patience, check_on_train_epoch_end=True, mode="min")],
                         callbacks=[pl.callbacks.EarlyStopping(monitor='validation_epoch_loss',
                                                               patience=args.patience,
                                                               check_on_train_epoch_end=True,
                                                               mode="min")],
                         # limit_val_batches=0.1,
                         num_sanity_val_steps=0,
                         deterministic="warn",
                         default_root_dir=args.ckpt_save_path)

    pl.seed_everything(args.random_seed, workers=True)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    # trainer.predict(model, datamodule=datamodule)

"Single main run"
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    # devices="auto",
    # devices=[0, 1, 2],
    # devices=[0, 3],
    devices = [0]
    main(devices=devices)
