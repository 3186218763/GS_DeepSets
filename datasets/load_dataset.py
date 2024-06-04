import torch
import torch.nn as nn
import torch.nn.functional as F
from jinja2 import optimizer
from torch.utils.data import DataLoader, ConcatDataset
from datasets.gnss_dataset import Gnss_Dataset
from nets.DeepSets import DeepSet_Only
from functools import partial
from utools.Mylog import logger

torch.set_printoptions(precision=8, sci_mode=False)


# torch默认的collate_fn要求每个样本的shape必须相同，
# 我们要填充False，并记录False的pad_mask位置给神经网络,设置最大行数是32
def collate_fn(batch, padding_columns=None):
    sorted_batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)
    features = [x['features'] for x in sorted_batch]

    # 截断或填充序列
    features_padded = []
    pad_mask = []  # 初始化一个空列表，用于存储 pad_mask 的每个样本的掩码
    for feat in features:
        if feat.shape[0] < 32:
            # 填充
            pad_length = 32 - feat.shape[0]
            padded_feat = F.pad(feat, (0, 0, 0, pad_length), value=0)  # 在序列末尾填充 0
            features_padded.append(padded_feat)
            # 创建 pad_mask
            mask = [[1] * feat.shape[1] for _ in range(feat.shape[0])]  # 初始化掩码为全 1
            mask += [[0] * feat.shape[1] for _ in range(pad_length)]  # 填充过的位置为 0
            pad_mask.append(mask)
        else:
            # 截断
            truncated_feat = feat[:32, :]
            features_padded.append(truncated_feat)
            # 创建 pad_mask
            mask = [[1] * feat.shape[1] for _ in range(32)]  # 截断后全部为 1
            pad_mask.append(mask)

    features_padded = torch.stack(features_padded)
    pad_mask = torch.BoolTensor(pad_mask)

    # 处理 padding_columns 参数
    if padding_columns is not None:
        for i in padding_columns:
            pad_mask[:, :, i] = False  # 指定列的掩码位置为 False

    correction = torch.stack([x['right_correction'].clone().detach().float() for x in sorted_batch])

    retval = {
        'right_correction': correction,
        'features': features_padded,
    }
    return retval, pad_mask


def collate_fa(batch, padding_columns=None):
    sorted_batch = sorted(batch, key=lambda x: x['features'].shape[0], reverse=True)
    features = [x['features'] for x in sorted_batch]

    # 截断或填充序列
    features_padded = []
    pad_mask = []  # 初始化一个空列表，用于存储 pad_mask 的每个样本的掩码
    for feat in features:
        if feat.shape[0] < 32:
            # 填充
            pad_length = 32 - feat.shape[0]
            padded_feat = F.pad(feat, (0, 0, 0, pad_length), value=0)  # 在序列末尾填充 0
            features_padded.append(padded_feat)
            # 创建 pad_mask
            mask = [[1] * feat.shape[1] for _ in range(feat.shape[0])]  # 初始化掩码为全 1
            mask += [[0] * feat.shape[1] for _ in range(pad_length)]  # 填充过的位置为 0
            pad_mask.append(mask)
        else:
            # 截断
            truncated_feat = feat[:32, :]
            features_padded.append(truncated_feat)
            # 创建 pad_mask
            mask = [[1] * feat.shape[1] for _ in range(32)]  # 截断后全部为 1
            pad_mask.append(mask)

    features_padded = torch.stack(features_padded)
    pad_mask = torch.BoolTensor(pad_mask)

    # 处理 padding_columns 参数
    if padding_columns is not None:
        for i in padding_columns:
            pad_mask[:, :, i] = False  # 指定列的掩码位置为 False

    correction = torch.stack([x['right_correction'].clone().detach().float() for x in sorted_batch])
    init = torch.stack([x['init_position'].clone().detach().float() for x in sorted_batch])
    real = torch.stack([x['real_position'].clone().detach().float() for x in sorted_batch])

    retval = {
        'right_correction': correction,
        'features': features_padded,
        'init_position': init,
        'real_position': real,
    }
    return retval, pad_mask


# 使用 functools.partial 传递 padding_columns 参数
collate_feat = partial(collate_fa)

if __name__ == '__main__':

    train_dir2 = "../data/train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4"
    dataset2 = Gnss_Dataset(train_dir2)

    loader = DataLoader(dataset2, batch_size=16, shuffle=True, collate_fn=collate_feat)

    for i, batch_sample in enumerate(loader):
        _batch_sample, pad_mask = batch_sample
        features = _batch_sample['features']
        right_correction = _batch_sample['right_correction']
        pad_mask = pad_mask
        print(pad_mask)
