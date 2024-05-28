import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from functools import partial
from utools.Mylog import logger
from datasets.gnss_dataset import Gnss_Dataset
from datasets.load_dataset import collate_fn
from nets.DeepSets import DeepSet_Only
from nets.DeepSet_Dense import DeepSet_Dense
from nets.DeepSet_ResNet import DeepSet_ResNet


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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    pre_dataset = Gnss_Dataset(pre_dir)
    collate_feat = partial(collate_fn, pad_columns=pad_columns)
    pre_loader = DataLoader(pre_dataset,
                            collate_fn=collate_feat,
                            drop_last=True,
                            batch_size=batch_size)

    if model_name == "DeepSet_Only":
        net = DeepSet_Only(input_size=6)
    elif model_name == "DeepSet_Dense":
        net = DeepSet_Dense(input_size=6)
    elif model_name == "DeepSet_ResNet":
        net = DeepSet_ResNet(input_size=6)
    elif model_name == "RNN":
        net = None
    else:
        msg = f"没有{model_name}这个模型"
        raise ValueError(msg)
    net.load_state_dict(model_parameter_dir)
    net.to(device)
    for i, batch_sample in enumerate(tqdm(pre_loader, desc="预测进度")):
        _batch_sample, pad_mask = batch_sample
        features = _batch_sample['features'].to(device)
        right_correction = _batch_sample['right_correction'].to(device)
        pad_mask = pad_mask.to(device)
        out = net(features, pad_mask=pad_mask)
        logger.info(
            f"第{i}组, right_correction:{right_correction}, out_correction:{out}, 偏远量:{out - right_correction}, 距离：{euclidean_distance(out, right_correction)}")
