import os
from argparse import ArgumentParser
import math
import ray
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch

"Don't touch the code below!"
def import_lightning():
    try:
        import lightning.pytorch as pl
    except ModuleNotFoundError:
        import pytorch_lightning as pl
    return pl
pl = import_lightning()

from data.MyDataset import *
from model.mycallback import MyEarlyStopping
from model.MyModel import *
import torch
from data.lightingdata import MyLigDataModule
import numpy as np
from ray import tune
# from ray.tune.search.ax import AxSearch
# from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, MedianStoppingRule, FIFOScheduler
# from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
### 这个TuneReportCheckpointCallback官方没更新，还用的是pytorch_lightning，而我的lighting2.0用的是pytorch.lightning
import ptwt, pywt
from functools import partial
from main import *
from main_sub import *
import shutil
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig



"""
定义主函数
"""

def main(Sample_config, args):
    args = update_args(Sample_config, args)
    print("请确认传感器数是否是：", args.sensor_num, "请确认节点数目即通道数是否是：", args.node_num)

    datamodule = MyLigDataModule(args)
    model = MyLigModel(args)
    # model = torch.compile(MyLigModel(args))
    # 暂不能用，会报错：https://github.com/Lightning-AI/pytorch-lightning/issues/18835
    # 而且我在MyModel.py里用了：self.MySTGNN = torch.compile(MySTGNN(self.args))

    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu",
        fast_dev_run=0,
        strategy="auto",
        # devices="auto",
        # accelerator="auto",
        # fast_dev_run=0,
        # strategy="auto",
        ## max_epochs=args.max_epoch,
        num_nodes=1,
        enable_progress_bar=True,
        # logger=pl.loggers.TensorBoardLogger(save_dir=args.ckpt_save_path, name="lightning_logs"),
        callbacks=[
            # pl.callbacks.EarlyStopping(
            #     monitor="training_loss", patience=args.patience,
            #     check_on_train_epoch_end=True, mode="min"),
            pl.callbacks.EarlyStopping(
                monitor='validation_epoch_loss', patience=args.patience,
                check_on_train_epoch_end=True, mode="min"),
        ],
        # limit_val_batches=0.1,
        num_sanity_val_steps=0,
        deterministic="warn",
        default_root_dir=args.ckpt_save_path,
        check_val_every_n_epoch=1,
    )

    pl.seed_everything(args.random_seed, workers=True)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    # trainer.predict(model, datamodule=datamodule)



"""
定义主函数，for ray-tune
"""

def light_main(Sample_config, args):
    args = update_args(Sample_config, args)
    print("请确认传感器数是否是：", args.sensor_num, "请确认节点数目即通道数是否是：", args.node_num)

    datamodule = MyLigDataModule(args)
    model = MyLigModel(args)
    # model = torch.compile(MyLigModel(args))
    # 暂不能用，会报错：https://github.com/Lightning-AI/pytorch-lightning/issues/18835
    # 而且我在MyModel里用了：self.MySTGNN = torch.compile(MySTGNN(self.args))

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        fast_dev_run=0,
        strategy=RayDDPStrategy(),
        ## max_epochs=args.max_epoch,
        enable_progress_bar=True,
        # logger=pl.loggers.TensorBoardLogger(save_dir=args.ckpt_save_path, name="lightning_logs"),
        callbacks=[
            RayTrainReportCallback(),
            # TuneReportCheckpointCallback(metrics=['training_loss'],on="train_end"),
            # TuneReportCheckpointCallback(
            #     metrics=['train_end_loss', 'ray_metric',
            #              'AD_F1', 'AD_precision', 'AD_recall', 'AD_AUC', 'AD_accuracy', 'AD_threshold', 'AD_normal_MSE', 'AD_normal_MAE',
            #              'FC_MSE', 'FC_MAE', 'FC_RMSE', 'FC_MAPE', 'FC_MSPE', 'FC_RSE', 'FC_CORR'],
            #     on="test_epoch_end"),
            # pl.callbacks.EarlyStopping(
            #     monitor="training_loss", patience=args.patience,
            #     check_on_train_epoch_end=True, mode="min"),
            pl.callbacks.EarlyStopping(
                monitor='validation_epoch_loss', patience=args.patience,
                check_on_train_epoch_end=True, mode="min"),
        ],
        plugins=[RayLightningEnvironment()],
        # limit_val_batches=0.1,
        num_sanity_val_steps=0,
        deterministic="warn",
        default_root_dir=args.ckpt_save_path,
        check_val_every_n_epoch=1,
    )

    pl.seed_everything(args.random_seed, workers=True)

    trainer = prepare_trainer(trainer)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    # trainer.predict(model, datamodule=datamodule)


# main(Search_config=None)

"""
定义ray tune调参函数
"""

def ray_tune_run(Search_config, args, everyexp_gpu, exp_num):

    """配置args"""
    args = args_update_ray(Search_config, args)

    """配置搜索空间"""
    # Search_config = Search_config

    scheduler = FIFOScheduler()

    # scheduler = ASHAScheduler(
    #     time_attr="training_iteration",
    #     # metric='training_loss',
    #     # metric='validation_epoch_loss',
    #     # mode="min",
    #     max_t=args.max_epoch,
    #     grace_period=args.trail_grace_period,
    #     reduction_factor=args.trail_reduction_factor)

    # scheduler = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     # metric='validation_epoch_loss',
    #     # metric='training_loss',
    #     # mode="min",
    #     perturbation_interval=args.patience,
    #     # burn_in_period=1,
    #     # hyperparam_mutations=Search_config,
    #     hyperparam_mutations={"train_loop_config": Search_config},
    #     # hyperparam_mutations={"lightning_config": Search_config},
    #     quantile_fraction=0.25,
    #     # resample_probability=0.25,
    #     resample_probability=1.0,
    # )

    # scheduler = MedianStoppingRule(
    #     time_attr="training_iteration",
    #     # metric='training_loss',
    #     metric='validation_epoch_loss',
    #     mode="min",
    #     grace_period=args.patience,
    #     min_samples_required=10,
    #     min_time_slice=10,
    #     hard_stop=True,
    # )


    """定义搜索算法,None表示使用默认的随机搜索算法"""
    algo = None
    # algo = AxSearch(metric='validation_epoch_loss', mode="min")
    # algo = AxSearch()
    # algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    # algo = HyperOptSearch(metric='validation_epoch_loss', mode="min")
    # algo = OptunaSearch(metric='validation_epoch_loss', mode="min")

    # """定义ray进度报告器 """
    # if args.TASK == 'anomaly_detection':
    #     reporter = CLIReporter(
    #         parameter_columns=list(Search_config.keys()),
    #         metric_columns=['train_end_loss', 'ray_metric',
    #                         'AD_F1', 'AD_precision', 'AD_recall', 'AD_AUC', 'AD_accuracy', 'AD_threshold',
    #                         'AD_normal_MSE', 'AD_normal_MAE'])
    # elif args.TASK == 'forecast':
    #     reporter = CLIReporter(
    #         parameter_columns=list(Search_config.keys()),
    #         metric_columns=['train_end_loss', 'ray_metric',
    #                         'FC_MSE', 'FC_MAE', 'FC_RMSE', 'FC_MAPE', 'FC_MSPE'])
    # elif args.TASK == 'reconstruct':
    #     reporter = CLIReporter(
    #         parameter_columns=list(Search_config.keys()),
    #         metric_columns=['train_end_loss', 'ray_metric',
    #                         'FC_MSE', 'FC_MAE', 'FC_RMSE', 'FC_MAPE', 'FC_MSPE'])

    """num_workers=1表示不要把一个实验分解多个GPU，不要改"""
    resources_per_worker = {"CPU": 72 // exp_num, "GPU": everyexp_gpu}
    resources_per_trial = {"cpu": 72 // exp_num, "gpu": everyexp_gpu}
    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker=resources_per_worker,
    )
    light_main2 = tune.with_parameters(light_main, args=args)
    light_main3 = tune.with_resources(light_main2, resources_per_trial)

    """定义运行配置"""
    run_config = RunConfig(
        name=args.exp_name,
        storage_path=args.ckpt_save_path + '/' + 'ray_logs',
        # progress_reporter=reporter,
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            # checkpoint_score_attribute='training_loss',
            checkpoint_score_attribute='validation_epoch_loss',
            checkpoint_score_order="min",
        ),
    )

    """封装"""
    ray_trainer = TorchTrainer(
        light_main3,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    """最终定义tuner"""
    tuner = tune.Tuner(
        ray_trainer,
        # param_space=Search_config,
        param_space={"train_loop_config": Search_config},
        # param_space={"train_loop_config": Search_config},
        tune_config=tune.TuneConfig(
            metric='validation_epoch_loss',
            # metric='training_loss',
            # metric='ray_metric',
            mode="min",
            # mode="max",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=args.grid_num_samples,
            reuse_actors=True,
            # reuse_actors=False
        ),
        run_config=run_config,
    )

    log_path = args.ckpt_save_path + '/' + 'ray_logs' + '/' + args.exp_name
    if os.path.exists(log_path):
        print("已经存在该实验的日志，将继续训练，但是请注意，搜索空间不会改变，依然是上次实验的搜空间")
        tuner = tune.Tuner.restore(
            path=log_path,
            trainable=ray_trainer,
            resume_unfinished=True,
            resume_errored=True,
            # param_space=Search_config,
            param_space={"train_loop_config": Search_config},
        )

    """开始搜索"""
    result = tuner.fit()
    print("Best hyperparameters found were: ", result.get_best_result())

    """用最佳参数跑一次"""
    best_config = result.get_best_result().config['train_loop_config']
    print("best_config is: ", best_config)
    main(best_config, args)
    print("all done!")
    ray.shutdown()
    # import atexit
    # atexit.register(ray.shutdown)

    return result



"""配置搜索空间"""
"""
# tune.choice表示搜索空间是一个列表，会从中随机选一个
# tune.loguniform表示搜索空间是一个对数均匀分布，会从中随机选一个
# tune.uniform表示搜索空间是一个均匀分布，会从中随机选一个
# tune.quniform表示搜索空间是一个均匀分布，会从中随机选一个，但是是按照给定的步长来选的，如tune.quniform(0, 2, 0.2)
# tune.grid_search表示搜索空间是一个列表，会从中按顺序选一个，指定的值保证被采样
"""



"## 实验892：'Typical_Nonlinear_Operators1'，G3CN+MutualInfo，不分解, 重建，V7.892"
Search_config = {
    ### 实验设置
    'Version': 'V7.892',
    # 'Method': 'MadjGCN',
    'Method': 'MadjGCN_Project',
    'data_name': 'Typical_Nonlinear_Operators1',
    'Decompose': 'None',
    # 'Decompose': 'WaveletPacket',
    # 'TASK': 'forecast',
    # 'TASK': 'anomaly_detection',
    'TASK': 'reconstruct',
    # 'BaseOn': 'forecast',
    'BaseOn': 'reconstruct',
    'channel_to_channel': 'M',
    # 'MS_which': tune.choice(['-1']),
    # exp_name 在main_sub.py里设置
    'reco_form': 'all_to_all',
    'if_time_dimOne': True,

    ### node_num相关策略制定
    # 'if_timestamp': False,
    'if_timestamp': True,
    # 'if_timestamp': tune.choice([True, False]),
    # 选了STL
    'STL_seasonal': 7,
    # 选了Wavelet
    'Wavelet_wave': 'db4',
    'Wavelet_level': 2,  # must be set

    ### 数据
    'Dataset': tune.sample_from(lambda config: config['train_loop_config']['data_name'] + '_Dataset'),
    'sensor_num': tune.sample_from(get_config_sensor_num),
    'node_num': tune.sample_from(get_config_node_num),
    'timestamp_dim': tune.sample_from(get_config_timestamp_dim),
    'lag': 96,
    # 'lag_step': 1,
    # 'label_len': 24,
    # 'pred_len': tune.choice([96, 192, 336, 720]),
    # 'pred_step': tune.choice([tune.sample_from(lambda config: config['train_loop_config']['lag']),
    #                           tune.sample_from(lambda config: 0.5*config['train_loop_config']['lag'])]),
    # # 这个pred_step只在forecast且pred_len大于lag时才需要设置，设置时一般就是lag和0.5*lag就好
    'num_workers': 3,
    'feature': 'M',
    # 'target': tune.choice(['OT']),

    ### 预处理
    'scale': True,
    'inverse': False,
    'preMA': False,
    # 'preMA_win': tune.choice([5]),

    ### 图结构建立
    'graph_ca_len': 3000,  # 这个设置大一些，因为MIC有大量数据
    # 'graph_ca_meth': tune.choice(['MIC', 'Copent']),
    'graph_ca_meth': 'MutualInfo',
    # 'graph_ca_thre': 0.6,
    'graph_ca_thre': tune.grid_search([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    # 如果选了MIC
    'MIC_alpha': 0.6,
    'MIC_c': 15,
    # 'MIC_c': tune.choice([15, 5]),

    ### 非线性
    'dropout': 0.1,
    'LeakyReLU_slope': 1e-3,

    ### 模型结构
    'Architecture': 'Series_TS',
    # 'Architecture': tune.choice(['Parallel', 'Series_ST', 'Series_TS', 'Series_STS']),
    # # 如果选了Parallel
    # 'fusion_method': tune.choice(['MLP_Concat', 'Gate_Weight', 'Add_Minus']),
    # 'fusion_hidden_dim': tune.choice([tune.sample_from(lambda config: 3*config['train_loop_config']['node_num']),
    #                                   tune.sample_from(lambda config: 6*config['train_loop_config']['node_num']),
    #                                   tune.sample_from(lambda config: 12*config['train_loop_config']['node_num'])]),

    # spatial_block, 【【【MAdjGCN就是G3CN， CMTS_GCN就是多层的G3CN】】】
    'spatial_method': 'MAdjGCN',
    # 'spatial_method': 'H2GCN',
    # 'spatial_method': 'GPRGNN',
    # 'spatial_method': 'None',
    # 'spatial_method': 'Muti_S_GAT',
    # 'spatial_method': 'GCN_s',
    ## 'spatial_method': 'CMTS_GCN',
    'self_edge': False,  # 自连接
    # 如果选了MAdjGCN
    'K': tune.grid_search([64, 128, 256]),
    # 'K': tune.choice([tune.sample_from(lambda config: 1 * config['train_loop_config']['node_num']),
    #                   tune.sample_from(lambda config: 2 * config['train_loop_config']['node_num']),
    #                   tune.sample_from(lambda config: 4 * config['train_loop_config']['node_num']),
    #                   tune.sample_from(lambda config: 8 * config['train_loop_config']['node_num'])]),
    'residual_alpha': 0.0,  # 空间卷积block内部
    'block_residual': 0,  # 空间卷积block之外
    # 'self_edge': tune.choice([True, False]),    # 自连接
    # # 如果选了CMTS_GCN，除了上面的参数，还有下面的参数
    # 'CMTS_GCN_K_nums': [tune.sample_from(lambda config: 3*config['train_loop_config']['node_num']) for _ in range(3)],
    # 'CMTS_GCN_residual': tune.choice([0, 0.2]),
    # # 如果选了GCN_s
    # 'GCN_layer_nums': tune.choice([...]),
    # 'self_edge': tune.choice([True, False]),    # 自连接图
    # 'block_residual': tune.choice([0, 1]),  # 空间卷积block之外
    # # 如果选了Muti_S_GAT
    # 'S_GAT_K': tune.choice([...]),
    # 'S_GAT_embed_dim': tune.choice([...]),
    # 'use_gatv2': tune.choice([...]),
    # 'block_residual': tune.choice([...]),  # 空间卷积block之外
    # # 如果选了GPRGNN
    # 'GPRGNN_K': tune.choice([...]),
    # # 如果选了H2GCN
    # 'H2GCN_embed_dim': tune.grid_search([...]),
    # # 'H2GCN_round_K': tune.choice([...]),
    # 'H2GCN_round_K': 3,

    # temporal_block
    'temporal_method': 'None',
    # 'temporal_method': 'GRU',
    # 'temporal_method': tune.choice(['TCN', 'GRU']),
    # 'temporal_method': tune.choice(['TCN', 'GRU', 'Muti_T_GAT']),
    # # 如果选了TCN
    # 'TCN_layers_channels': tune.choice([...]),
    # 'TCN_kernel_size': 2,
    # # 如果选了GRU
    # 'GRU_layers': tune.choice([...]),
    # 'GRU_hidden_num': tune.choice([...]),
    # # 如果选了Muti_T_GAT
    # 'use_gatv2': True,
    # 'T_GAT_K': tune.choice([...]),
    # 'T_GAT_embed_dim': tune.choice([...]),

    ### 优化器
    # 'optimizer': tune.choice([torch.optim.Adam]),
    'lr': tune.grid_search([0.001, 0.005, 0.01]),
    # 'lr': tune.choice([0.001, 0.005, 0.01]),
    # 'scheduler': tune.grid_search(['StepLR', 'ReduceLROnPlateau']),
    'scheduler': 'ReduceLROnPlateau',

    ### 训练配置
    'batch_size': 64,
    # 'random_seed': 42,
    # 'max_epoch': 500,
    'patience': 5, # 这个必须设置，连续5个epoch内loss都没有减下去就停止训练

    ### ray配置
    'trail_grace_period': 20,    # 最少几个epoch才能早停一个trail
    'trail_reduction_factor': 3,
    'grid_num_samples': 3,  # 这个必须设置，从search—space里采样多少组参数
}
devices = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = devices
all_exp_num = 80
args = set_args()
ray_tune_run(Search_config, args, everyexp_gpu=(math.ceil(len(devices)/2)/all_exp_num), exp_num=all_exp_num)




