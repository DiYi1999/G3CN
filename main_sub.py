

def set_node_num(data_name, Decompose=None, Wavelet_level=None, if_timestamp=False, reco_form=False):
    """

    Args:
        data_name:
        Decompose:
        Wavelet_level:
        if_timestamp:
        reco_form:

    Returns:
        sensor_num:
        node_num:
        timestamp_dim:

    """
    if data_name == 'ETTh1' or data_name == 'ETTh2':
        sensor_num = 7
        timestamp_dim = 4
    elif data_name == 'ETTm1' or data_name == 'ETTm2':
        sensor_num = 7
        timestamp_dim = 5
    elif data_name == 'weather':
        sensor_num = 21
        timestamp_dim = 5
    elif data_name == 'Electricity':
        sensor_num = 321
        timestamp_dim = 1
    elif data_name == 'exchange_rate':
        sensor_num = 8
        timestamp_dim = 1
    elif data_name == 'traffic':
        sensor_num = 862
        timestamp_dim = 4
    elif data_name == 'solar_energy':
        sensor_num = 137
        timestamp_dim = 1

    elif data_name == 'MIC_simulate':
        sensor_num = 13
        timestamp_dim = 1

    elif data_name == 'Typical_Nonlinear_Operators1':
        sensor_num = 13
        timestamp_dim = 1

    elif data_name == 'SixD_Hyperchaotic2':
        sensor_num = 6 if reco_form=='half_to_half' else 12
        timestamp_dim = 1

    elif data_name == 'Cart_Pendulum2':
        sensor_num = 4 if reco_form=='half_to_half' else 8
        timestamp_dim = 1

    elif data_name == 'Super_Nonlinear_Dataset22':
        sensor_num = 63
        timestamp_dim = 1

    else:
        raise ValueError('node_num is not defined， 请在main.py中定义node_num')

    if Decompose is not None:
        if Decompose == 'STL':
            node_num = sensor_num * 3
        elif Decompose == 'Wavelet':
            node_num = sensor_num * (Wavelet_level + 1)
        elif Decompose == 'WaveletPacket':
            node_num = sensor_num * (2**Wavelet_level)
        else:
            node_num = sensor_num

    if if_timestamp:
        node_num = node_num + timestamp_dim

    return sensor_num, node_num, timestamp_dim



def get_plot_pram(args):
    data_name = args.data_name
    if data_name == 'XJTU_SPS_1Hz_2024y9m18d':
        plot_pram = {'figsize': (4, 1.5), 'dpi': 1200, 'fontsize': 12}
    elif data_name == 'XJTU_SPS_01Hz_2024y9m19d':
        plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}
    elif data_name == 'XJTU_SPS_05Hz_2024y9m20d':
        plot_pram = {'figsize': (4, 1.5), 'dpi': 1200, 'fontsize': 12}
    elif data_name.startswith('SixD_Hyperchaotic'):
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    elif data_name.startswith('Double_2D_Spring'):
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    elif data_name.startswith('Cart_Pendulum'):
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    elif data_name.startswith('Physical_System_Synthetic'):
        plot_pram = {'figsize': (4, 2), 'dpi': 1200, 'fontsize': 12}
    else:
        if args.TASK == 'reconstruct':
            plot_pram = None
        elif args.TASK == 'forecast':
            plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}
        elif args.TASK == 'anomaly_detection':
            plot_pram = {'figsize': (6, 2), 'dpi': 1200, 'fontsize': None}
        else:
            plot_pram = {'figsize': (6, 1.5), 'dpi': 1200, 'fontsize': None}

    return plot_pram




def update_args(Sample_config, args):
    """
    更新args

    Args:
        Sample_config: 
        args:

    Returns:
        args: 更新后的args

    """
    # "# node_num相关参数可能在Search_config中没有定义，需要重新计算并更新"
    # sensor_num, node_num, timestamp_dim = set_node_num(Sample_config['data_name'],
    #                                                    Sample_config['Decompose'],
    #                                                    Sample_config['Wavelet_level'],
    #                                                    Sample_config['if_timestamp'],
    #                                                    Sample_config['reco_form'])
    # setattr(args, 'sensor_num', sensor_num)
    # setattr(args, 'node_num', node_num)
    # setattr(args, 'timestamp_dim', timestamp_dim)
    # setattr(args, 'Dataset', Sample_config['data_name'] + '_Dataset')

    "# 更新exp_name和data_path和save_path"
    exp_name = (Sample_config['Version'] + '_' + Sample_config['Method'] + '_' + Sample_config['data_name']
                + '_' + Sample_config['Decompose'] + '_' + Sample_config['TASK'])
    setattr(args, 'exp_name', exp_name)
    if 'data_path' not in Sample_config.keys():
        data_path = Sample_config['TASK'] + '/' + Sample_config['data_name'] + '/' + Sample_config['data_name']
    else:
        data_path = Sample_config['data_path']
    setattr(args, 'data_path', data_path)
    # data_path = Sample_config['TASK'] + '/' + Sample_config['data_name'] + '/' + Sample_config['data_name']
    # setattr(args, 'data_path', data_path)
    save_path = (args.result_root_path + '/' + Sample_config['Method']
                 + '/' + Sample_config['Version'] + '/' + Sample_config['data_name']
                 + '/' + exp_name)
    setattr(args, 'save_path', save_path)
    setattr(args, 'ckpt_save_path', save_path + '/ckpt')
    setattr(args, 'table_save_path', save_path + '/table')
    setattr(args, 'plot_save_path', save_path + '/plot')

    for key, value in Sample_config.items():
        setattr(args, key, value)

    return args


def args_update_ray(Search_config, args):

    variables = {'TASK': 'reconstruct', 'data_name': 'MIC_simulate', 'Decompose': 'None', 'Version': 'Vtest',
                 'Method': 'MadjGCN_Project', 'patience': 20, 'grid_num_samples': 1000}
    for key in variables.keys():
        if type(Search_config[key]) == str:
            variables[key] = Search_config[key]
        elif type(Search_config[key]) == int:
            variables[key] = Search_config[key]
        else:
            variables[key] = Search_config[key].categories[0]
    TASK, data_name, Decompose, Version, Method, patience, grid_num_samples = variables['TASK'], \
        variables['data_name'], variables['Decompose'], variables['Version'], variables['Method'], \
        variables['patience'], variables['grid_num_samples']

    exp_name = (Version + '_' + Method + '_' + data_name + '_' + Decompose + '_' + TASK)
    setattr(args, 'exp_name', exp_name)
    save_path = (args.result_root_path + '/' + Method + '/' + Version + '/' + data_name + '/' + exp_name)
    setattr(args, 'save_path', save_path)
    setattr(args, 'ckpt_save_path', save_path + '/ckpt')

    setattr(args, 'TASK', TASK)
    setattr(args, 'patience', patience)
    setattr(args, 'grid_num_samples', grid_num_samples)

    return args


def get_config_sensor_num(config):
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].reco_form)
    return sensor_num


def get_config_node_num(config):
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].reco_form)
    return node_num


def get_config_timestamp_dim(config):
    sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config']['data_name'],
                                                       config['train_loop_config']['Decompose'],
                                                       config['train_loop_config']['Wavelet_level'],
                                                       config['train_loop_config']['if_timestamp'],
                                                       config['train_loop_config']['reco_form'])
    # sensor_num, node_num, timestamp_dim = set_node_num(config['train_loop_config'].data_name,
    #                                                    config['train_loop_config'].Decompose,
    #                                                    config['train_loop_config'].Wavelet_level,
    #                                                    config['train_loop_config'].if_timestamp,
    #                                                    config['train_loop_config'].reco_form)
    return timestamp_dim
