import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from functools import partial
from utools.Mylog import logger
from datasets.gnss_dataset import Gnss_Dataset
from utools.DataSet_Tools import check_dataset
from datasets.load_dataset import collate_fn
from utools.ConfigManager import ConfigManager
from nets.DeepSets import DeepSet_Only
from nets.DeepSet_Dense import DeepSet_Dense
from nets.DeepSet_ResNet import DeepSet_ResNet
from nets.DeepSet_Snapshot import DeepSet_Snapshot


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def Predict(pre_dir, model_name, model_parameter_dir, pad_columns=None, batch_size=1):
    """
    Predict功能
    :param batch_size: 每次处理数据的batch_size，默认为1
    :param pad_columns: 这个默认是None，如果需要屏蔽某列的输入请设置
    :param pre_dir: 预测数据集的位置
    :param model_name: 使用的模型名称请使用str，比如"DeepSet_Only"
    :param model_parameter_dir: 模型参数位置
    :return:这个函数没有返回值
    """
    configs = ConfigManager()
    configs.set_default_config(model_name)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    pre_dataset = Gnss_Dataset(pre_dir, keep_init_real=True)
    collate_feat = partial(collate_fn, pad_columns=pad_columns)
    pre_loader = DataLoader(pre_dataset,
                            collate_fn=collate_feat,
                            drop_last=True,
                            batch_size=batch_size)
    check_dataset(pre_loader)
    if configs.model_name == "DeepSet_Only":
        net = DeepSet_Only(input_size=configs.features_len,
                           output_size=configs.output_size,
                           hidden_size=configs.hidden_size,
                           Debug=configs.Debug)

    elif configs.model_name == "DeepSet_Dense":
        net = DeepSet_Dense(deepset_hidden_size=configs.deepset_hidden_size,
                            deepset_out_size=configs.deepset_out_size,
                            Debug=configs.Debug)

    elif configs.model_name == "DeepSet_ResNet":
        net = DeepSet_ResNet(input_size=configs.features_len,
                             deepset_hidden_size=configs.deepset_hidden_size,
                             deepset_out_size=configs.deepset_out_size,
                             Debug=configs.Debug)

    elif configs.model_name == "DeepSet_Snapshot":
        net = DeepSet_Snapshot(input_size=configs.features_len,
                               deepset_hidden_size=configs.deepset_hidden_size,
                               deepset_out_size=configs.deepset_out_size,
                               Debug=configs.Debug)

    else:
        raise ValueError("没有此模型，请查看可用的模型配置")

    state_dict = torch.load(model_parameter_dir)
    net.load_state_dict(state_dict)
    net.to(device)

    for i, batch_sample in enumerate(pre_loader):
        _batch_sample, pad_mask = batch_sample
        features = _batch_sample['features'].to(device)
        right_correction = _batch_sample['right_correction'].to(device)
        init_position = _batch_sample['init_position'].to(device)
        pad_mask = pad_mask.to(device)
        out = net(features, pad_mask=pad_mask)
        guess_position = init_position + out
        real_position = _batch_sample['real_position'].to(device)
        guess_position = guess_position.cpu().numpy()
        real_position = real_position.cup().numpy()



