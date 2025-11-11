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
### 这个TuneReportCheckpointCallback官方没更新，还用的是pytorch_lightning，而我的lighting2.0用的是pytorch.lightning
import ptwt, pywt
from main_sub import *


def set_args():
    parser = ArgumentParser()

    ### 任务设置
    parser.add_argument('--TASK', type=str, default='forecast',
                        help='anomaly_detection or forecast or reconstruct'
                             '此处TASK和下面的BaseOn的关系是：'
                             '如果TASK是anomaly_detection，完成异常检测任务可以是基于重建，也可以是基于预测，'
                             '如果TASK是forecast，那么BaseOn只能是forecast，不需要label数据文件'
                             '如果TASK是reconstruct，那么BaseOn只能是reconstruct，不需要label数据文件只重建')
    parser.add_argument('--BaseOn', type=str, default='forecast',
                        help='reconstruct or forecast，默认异常检测也是基于预测的')
    parser.add_argument('--data_name', type=str, default='ETTh1', help='数据集名字')
    parser.add_argument('--Decompose', type=str, default='None', help='None 不分解/STL/WaveletPacket/Wavelet')
    # if TASK == 'forecast':
    parser.add_argument('--channel_to_channel', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'MS:multivariate predict univariate, S:univariate predict univariate')
    parser.add_argument('--MS_which', type=int, default=1, help='如果上面选了MS，那么要指定用哪个单列')
    # if TASK == 'reconstruct':
    parser.add_argument('--reco_form', type=str, default='all_to_all', help='options:[all_to_all, half_to_half]')
    parser.add_argument('--if_time_dimOne', type=bool, default=True,
                        help='数据集时间是否是乱的，若为True，那么默认数据集的第一维是t时间标签，这第一维将不参与重建，而是被用来画图作为横轴，不然画出的图是乱的')



    ### 实验设置
    parser.add_argument('--Version', type=str, default='V1.0', help='代码版本 V1.0')
    parser.add_argument('--Method', type=str, default='MadjGCN_Project', help='项目名称，最大一级文件目录')
    exp_name = parser.parse_known_args()[0].Version + '_' + \
               parser.parse_known_args()[0].Method + '_' + \
               parser.parse_known_args()[0].data_name + '_' + \
               parser.parse_known_args()[0].Decompose + '_' + \
               parser.parse_known_args()[0].TASK
    parser.add_argument('--exp_name', type=str, default=exp_name, help='实验名称/excel文件名/保存文件夹名等')



    ### 数据导入路径
    """f = open(os.path.join(self.root_path, self.data_path, '{}_train.csv'.format(self.data_path)), "rb")"""

    parser.add_argument('--root_path', type=str, default='/home/data/DiYi/DATA',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=parser.parse_known_args()[0].TASK + '/' +
                                                         parser.parse_known_args()[0].data_name + '/' +
                                                         parser.parse_known_args()[0].data_name,
                        help='ETTh1/ETTh1')



    ### 路径相关
    parser.add_argument('--result_root_path', type=str, default='/home/data/DiYi/MyWorks_Results',
                        help='实验结果的保存路径')
    save_path = parser.parse_known_args()[0].result_root_path + '/' + \
                parser.parse_known_args()[0].Method + '/' + \
                parser.parse_known_args()[0].Version + '/' + \
                parser.parse_known_args()[0].data_name + '/' + \
                parser.parse_known_args()[0].exp_name
    parser.add_argument('--ckpt_save_path', type=str, default= save_path + '/ckpt',
                        help='检查点缓存路径，同时也是日志保存路径')
    parser.add_argument('--table_save_path', type=str, default=save_path + '/table',
                        help='调试结果的csv/excel文件保存路径')
    parser.add_argument('--plot_save_path', type=str, default=save_path + '/plot',
                        help='画图的保存路径')



    ### node_num相关策略制定
    # 如果上面的Decompose不是None，STL/Wavelet分解参数
    parser.add_argument('--STL_seasonal', type=int, default=7,
                        help='Length of the seasonal smoother. Must be an odd integer, and should normally be >= 7 (default).')
    parser.add_argument('--Wavelet_wave', type=str, default='db4', help='小波基函数，默认db4')
    parser.add_argument('--Wavelet_level', type=int, default=2, help='小波分解层数')
    parser.add_argument('--if_timestamp', type=bool, default=True, help='是否在预测或重建时利用上timestamp数据')

    sensor_num, node_num, timestamp_dim = set_node_num(parser.parse_known_args()[0].data_name,
                                                       parser.parse_known_args()[0].Decompose,
                                                       parser.parse_known_args()[0].Wavelet_level,
                                                       parser.parse_known_args()[0].if_timestamp,
                                                       parser.parse_known_args()[0].reco_form)



    ### 数据
    parser.add_argument('--Dataset', default= parser.parse_known_args()[0].data_name + '_Dataset',
                        help='使用MyDataset文件里编写的哪个Dataset')
    parser.add_argument('--dataset_split_ratio', type=int, default=0.8, help='数据集划分比例，训练集和验证集占比')

    parser.add_argument('--sensor_num', type=int, default=sensor_num, help='sensor_num')
    parser.add_argument('--node_num', type=int, default=node_num, help='node_num')
    parser.add_argument('--timestamp_dim', type=int, default=timestamp_dim,
                        help='dataset会另外返回timestamp，存储年、月、天、时分秒等')
    # 滑窗和滑窗步
    parser.add_argument('--lag', type=int, default=24, help='lag，即seq_len/slide_win滑窗长')  #
    lag = parser.parse_known_args()[0].lag
    parser.add_argument('--lag_step', type=int, default=1, help='那些数据集的 滑窗在取样本时 几步几步地滑，默认为1')  #
    # if TASK == 'forecast':
    parser.add_argument('--label_len', type=int, default=1, help='如果是预测任务，Transformer类模型需要提示片段，其他模型不需要')  #
    parser.add_argument('--pred_len', type=int, default=24, help='如果是预测任务，预测几步')
    parser.add_argument('--pred_step', type=int, default=96,
                        help='该参数只在预测长度pred_len大于输入时间步长度lag时发挥作用，因为需要多次循环RNN以生成足够长的pred_len'
                             '每次循环预测pred_step时间步，具体来说，pred_len除pred_step向上取整就是循环次数，默认值和lag保持一致，或取lag的一半')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--features', type=str, default='M', help='[S, M],前者是只用部分传感器的数据，M是用全部')
    parser.add_argument('--target', type=str, default='OT', help='若S，选择要用的传感器/特征')
    # parser.add_argument('--missing_rate', type=float, default=0, help='数据缺失率')  #
    # parser.add_argument('--missvalue', default=np.nan, help='人为制造缺失的时候缺失位置补np.nan还是0')



    ### 预处理
    parser.add_argument('--scale', type=bool, default=True, help='是否对数据进行标准化归一化')
    parser.add_argument('--inverse', type=bool, default=False, help='test完成后是否逆标准化进行恢复')
    # parser.add_argument('--preIDW', type=bool, default=True, help='是否对缺失的数据进行IDW插值 的预处理')
    parser.add_argument('--preMA', type=bool, default=False, help='是否对含噪数据进行滑动平均 的预处理')
    parser.add_argument('--preMA_win', type=int, default=5, help='滑动平均窗口大小')



    ### 图结构建立
    parser.add_argument('--graph_ca_len', type=int, default=1000,
                        help='建图时，读取多长训练数据段进行计算，这个设置大一些，因为MIC有大量数据才够，5000就挺好')
    parser.add_argument('--graph_ca_meth', type=str, default='MIC',
                        help='MIC：最大信息系数 / Copent：Copent熵 / Cosine：余弦相似度')
    parser.add_argument('--graph_ca_thre', type=float, default=0.6,
                        help='建图时，MIC/Copula阈值选择依据，大于阈值的边保留')
    parser.add_argument('--self_edge', type=bool, default= False,
                        help='是否使用自连接图，这个功能与下面的残差功能有一定重复，选择性使用')
    parser.add_argument('--graph_if_norm_A', type=bool, default=False,
                        help='建图时，是否对邻接矩阵进行归一化处理，默认False,H2GCN必须是False,但GPRGNN、GCN必须是True')
    # MIC参数
    parser.add_argument('--MIC_alpha', type=float, default=0.6, help='MIC参数alpha')
    parser.add_argument('--MIC_c', type=int, default=15, help='MIC参数c')



    ### 各method共享的参数
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--LeakyReLU_slope', type=float, default=1e-3, help='LeakyReLU激活函数的负斜度，默认1e-2，如果0则是ReLU')



    ### 如果args.Method选择了OursMethod：
    parser.add_argument('--Architecture', type=str, default='Series_TS', help='Parallel/Series_ST/Series_TS/Series_STS')

    ## if Parallel
    parser.add_argument('--fusion_method', type=str, default='MLP_Concat', help='MLP_Concat/Gate_Weight/Add_Minus')
    parser.add_argument('--fusion_hidden_dim', type=int, default=node_num * 6, help='MLP_Concat的隐藏层维度')

    ## Spatial_block
    parser.add_argument('--spatial_method', type=str, default='MAdjGCN',
                        help='MAdjGCN/CMTS_GCN/MAdjGCN_Lite/GCN_s/Muti_S_GAT/GPRGNN/H2GCN/None' \
                        '【【【MAdjGCN就是G3CN， CMTS_GCN就是多层的G3CN】】】')
    parser.add_argument('--block_residual', type=float, default=0,
                        help='在空间卷积block之前后，是否 残差连接 的系数，默认0/1就行')
    # if MAdjGCN or MAdjGCN_Lite ，MAdjGCN就是G3CN
    parser.add_argument('--K', type=int, default=node_num * 4, help='number of AdjGCN，根据UAT，得足够大！试试32，64，128，256')
    parser.add_argument('--residual_alpha', type=float, default=0,
                        help='在空间卷积block内部，在进入激活函数前是否残差连接，的系数，默认0')
    # if CMTS_GCN，CMTS_GCN就是多层的G3CN
    parser.add_argument('--CMTS_GCN_K_nums', type=list, default=[node_num*3, node_num*3],
                        help='CMTS_GCN是多层的MAdjGCN，这个是每层的隐藏神经元个数K值组成列表，' \
                        '根据UAT，加深网络深度有益于减轻网络宽度负担，试试[64,64]， [128,128]， [256,256]，甚至更多层[256,256,256]等')
    parser.add_argument('--CMTS_GCN_residual', type=float, default=0,
                        help='在CMTS_GCN的layers之间是否连接残差，的系数，默认0')
    # if GCN_s
    parser.add_argument('--GCN_layer_nums', type=list, default=[lag, lag],
                        help='比如[64, 64]，表示有两层GCN，每层隐藏神经元个数64，' \
                             '注意，如果是预测任务，GCN_layer_nums[-1]不能小于pred_len')
    # if Muti_S_GAT
    parser.add_argument('--S_GAT_K', type=int, default=3, help='几个GAT头，默认1')
    parser.add_argument('--S_GAT_embed_dim', type=int, default=96, help='GAT用于计算权重的embed_dim')
    # if GIN
    parser.add_argument('--GIN_layer_nums', type=list, default=[lag*2, lag*2], help='比如[32,16,8]')
    parser.add_argument('--GIN_MLP_layer_num', type=int, default=1, help='每层GIN 公式里的MLP的层数')
    # if SGC
    parser.add_argument('--SGC_hidden_dim', type=int, default=int(lag*0.5), help='SGC的隐藏层维度')
    parser.add_argument('--SGC_K', type=int, default=3, help='SGC的K值')
    # if GPRGNN
    parser.add_argument('--GPRGNN_K', type=int, default=3, help='GPRGNN的K值，默认3')
    # if H2GCN
    parser.add_argument('--H2GCN_embed_dim', type=int, default=int(lag * 2), help='H2GCN的编码特征长度')
    parser.add_argument('--H2GCN_round_K', type=int, default=3, help='H2GCN的拼接round')

    ## Temporal_block
    parser.add_argument('--temporal_method', type=str, default='GRU', help='TCN/GRU/Muti_T_GAT/None')
    # if TCN
    parser.add_argument('--TCN_layers_channels', type=list, default=[node_num, node_num],
                        help='比如[24, 8, 1]，代表3层，各层通道数从27依次变化为24变8变1。')
    parser.add_argument('--TCN_kernel_size', type=int, default=2, help='TCN的卷积核大小，默认2')
    # if GRU
    parser.add_argument('--GRU_layers', type=int, default=1, help='how many GRU')
    parser.add_argument('--GRU_hidden_num', type=int, default=64, help='GRU的h的维度，默认就等于传感器数')
    # if Muti_T_GAT
    parser.add_argument('--use_gatv2', type=bool, default=True, help='是使用GATV2还是GAT，GATV2是GAT的改进版')
    parser.add_argument('--T_GAT_K', type=int, default=3, help='几个GAT头，默认1')
    parser.add_argument('--T_GAT_embed_dim', type=int, default=parser.parse_known_args()[0].lag,
                        help='GAT用于计算权重的embed_dim')



    ### 异常分数计算
    parser.add_argument('--how_precision', type=bool, default=False, help='Deprecated parameter')
    parser.add_argument('--focus_on', type=str, default='F1', help='Deprecated parameter')
    parser.add_argument('--S_moving_average_window', type=int, default=3, help='Deprecated parameter')




    ### 优化器
    parser.add_argument('--optimizer', default=torch.optim.Adam, help='使用的优化器，默认torch.optim.Adam')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', type=str,
                        help='使用哪种学习率衰减策略，ReduceLROnPlateau、StepLR、ExponentialLR、CosineAnnealingLR、CosineAnnealingWarmRestarts')



    ### 训练配置
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--patience', type=int, default=5, help='连续10个epoch内loss都没有减下去就停止训练')
    parser.add_argument('--max_epoch', type=int, default=500, help='max epoch')



    ### Ray Tune
    # 超参搜索计划ASHAScheduler, Trial 是一次尝试
    # parser.add_argument('--max_trail', type=int, default=500, help='最大的trail数量')
    parser.add_argument('--trail_grace_period', type=int, default=40,
                        help='开始减少trial的数量之前要运行的最小trial数量')
    # parser.add_argument('--trail_time_out', type=int, default=600, help='trial运行的最大时间（秒）')
    parser.add_argument('--trail_reduction_factor', type=int, default=3,
                        help='reduction_factor表示每次减少trial数量的比例，'
                             '见https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/')
    parser.add_argument('--grid_num_samples', type=int, default=100, help='如果是-1，会无限重复搜索，直到达到停止条件')


    # parser.set_defaults(max_epochs=100)
    args = parser.parse_args()

    return args




"""
定义主函数,单独运行版
"""
def main(devices):
    args = set_args()
    print("请确认传感器数是否是：", args.sensor_num, "请确认节点数目即通道数是否是：", args.node_num)

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

"单个main运行"
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    # devices="auto",
    # devices=[0, 1, 2],
    # devices=[0, 3],
    devices = [0]
    main(devices=devices)













