from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR

from nets.DeepSets import DeepSet_Only
from nets.DeepSet_Dense import DeepSet_Dense
from nets.DeepSet_ResNet import DeepSet_ResNet
from nets.DeepSet_Snapshot import DeepSet_Snapshot

from utools.Train_Tools import find_train_dirs
from utools.Net_Tools import gradient_hook
from utools.DataSet_Tools import check_dataset
from utools.ConfigManager import ConfigManager

from datasets.load_dataset import collate_fa
from datasets.gnss_dataset import Gnss_Dataset

import numpy as np

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def Train_Model(base_dir, model_config_name, model_args_path=None):
    """
    训练指定的model
    :param model_args_path: 模型参数路径
    :param base_dir: 训练的dataset的路径
    :param model_config_name: 不需要填路径，直接填名字就可以了
    """

    # 创建config_manager，指定模型的配置文件
    configs = ConfigManager()
    configs.set_default_config(model_config_name)

    # 确保保存模型的文件夹存在，如果不存在则创建
    root_save_dir = "../model_saves/"
    root_save_dir = os.path.join(root_save_dir, configs.model_name)
    if not os.path.exists(root_save_dir):
        os.makedirs(root_save_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    collate_feat = partial(collate_fa, padding_columns=configs.padding_columns)
    # dataset可以设置保留features不同列，默认列顺序是pr(),prr(),los_vector(归一化后卫星到用户方向，3列),len(卫星到用户的距离)
    print("正在加载train_dataset")
    train_dirs = find_train_dirs(base_dir)
    dataset_list = []
    for train_dir in train_dirs:
        dataset = Gnss_Dataset(train_dir, debug=configs.Debug, normalize=True, keep_init_real=configs.keep_init_real)
        dataset_list.append(dataset)

    train_dataset = ConcatDataset(dataset_list)
    train_loader = DataLoader(train_dataset,
                              batch_size=configs.batch_size,
                              shuffle=configs.Is_shuffle,
                              num_workers=configs.num_workers,
                              drop_last=configs.Is_drop_last,
                              collate_fn=collate_feat)

    if configs.check_dataset:
        check_dataset(train_loader)

    # 设置训练的model
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

    # 设置loss_fn,优化器，和迭代器，把模型net放入device中
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=configs.learning_rate, eps=configs.eps)
    scheduler = StepLR(optimizer, step_size=configs.step_size, gamma=configs.gamma)
    losses = []
    net.to(device)
    if model_args_path is not None:
        state_dict = torch.load(model_args_path)
        net.load_state_dict(state_dict)
        print("模型参数加载成功")


    # 设置是否Debug，可以记录模型梯度的变化，排除异常
    if configs.Debug:
        net.register_full_backward_hook(gradient_hook)

    # 训练部分
    for epoch in tqdm(range(configs.epochs), desc="Epoch"):
        total_loss = 0.0
        for i, batch_sample in enumerate(train_loader):
            _batch_sample, pad_mask = batch_sample
            features = _batch_sample['features'].to(device)
            right_correction = _batch_sample['right_correction'].to(device)
            pad_mask = pad_mask.to(device)

            out = net(features, pad_mask=pad_mask)
            loss = loss_fn(out, right_correction)
            print(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 每个 epoch 结束后打印总体损失
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        tqdm.write(f"Epoch [{epoch + 1}/{configs.epochs}], train_Loss: {avg_loss}")

        # 每 5 个 epoch 结束后保存模型参数
        # 每 5 个 epoch 结束后保存模型参数，并绘制损失曲线
        if (epoch + 1) % 5 == 0:
            # 创建文件夹路径并确保存在
            save_dir = os.path.join(root_save_dir, f"epoch_{epoch + 1}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存模型参数
            torch.save(net.state_dict(), os.path.join(save_dir, "model.pt"))

            # 绘制损失曲线
            plt.figure()
            plt.plot(range(1, epoch + 2), losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.savefig(os.path.join(save_dir, "loss_curve.png"))
            plt.close()


if __name__ == '__main__':
    base_dir = "../data/train_2"
    model_config_name = "DeepSet_Only.yaml"
    model_args_path = None
    Train_Model(base_dir, model_config_name, model_args_path)
