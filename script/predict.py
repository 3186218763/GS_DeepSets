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


def euclidean_distance(coord1, coord2):
    """
    计算两个三维坐标之间的欧氏距离
    :param coord1: 第一个坐标 (x, y, z)
    :param coord2: 第二个坐标 (x, y, z)
    :return: 距离 (米)
    """
    return np.sqrt(np.sum((coord1 - coord2) ** 2))


def calc_score(llh, llh_gt):
    """
    计算预测位置和实际位置之间的评分
    :param llh: 预测的地面位置坐标，形状为 (n, 3) 的 numpy 数组
    :param llh_gt: 实际的地面位置坐标，形状为 (n, 3) 的 numpy 数组
    :return: 全部得分
    """
    d = np.array([euclidean_distance(p, gt) for p, gt in zip(llh, llh_gt)])
    return d


def Predict(base_dir, model_args_path=None):
    """
    训练指定的model
    :param model_args_path: 模型参数路径
    :param base_dir: 预测的dataset的路径
    """

    # 确保保存模型的文件夹存在，如果不存在则创建

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


    elif configs.model_name == "DeepSet_Snapshot":
        net = DeepSet_Snapshot(input_size=configs.features_len,
                               deepset_hidden_size=configs.deepset_hidden_size,
                               deepset_out_size=configs.deepset_out_size,
                               Debug=configs.Debug)

    else:
        raise ValueError("没有此模型，请查看可用的模型配置")

    net.to(device)
    if model_args_path is not None:
        state_dict = torch.load(model_args_path)
        net.load_state_dict(state_dict)
        print("模型参数加载成功")

    # 设置是否Debug，可以记录模型梯度的变化，排除异常
    if configs.Debug:
        net.register_full_backward_hook(gradient_hook)

    guess_positions = []
    real_positions = []
    init_positions = []
    # 预测部分
    for i, batch_sample in enumerate(train_loader):
        _batch_sample, pad_mask = batch_sample
        features = _batch_sample['features'].to(device)
        init_position = _batch_sample['init_position'].to(device)
        real_position = _batch_sample['real_position'].to(device)
        pad_mask = pad_mask.to(device)

        out = net(features, pad_mask=pad_mask)

        guess_position = init_position + out
        guess_positions.append(guess_position.cpu().detach())
        real_positions.append(real_position.cpu().detach())
        init_positions.append(init_position.cpu().detach())

        # 将列表中的张量拼接为一个张量
    guess_positions = torch.cat(guess_positions).numpy()
    real_positions = torch.cat(real_positions).numpy()
    init_positions = torch.cat(init_positions).numpy()

    return guess_positions, real_positions, init_positions


def plot_and_save_direction_bar(index, direction, init_positions, real_positions, guess_positions):
    # Calculate average errors
    init_error = np.mean(np.abs(init_positions[index] - real_positions[index]))
    guess_error = np.mean(np.abs(guess_positions[index] - real_positions[index]))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(['Initial', 'Corrected (Guess)'], [init_error, guess_error], color=[init_color, guess_color])

    plt.xlabel('Position Type')
    plt.ylabel(f'Error: (m)')
    plt.title(f'{direction}-Direction Average Distance Error Comparison')

    # Save the figure
    plt.savefig(f'../model_load/{configs.model_name}/{direction}_bar.png')

    # Show the figure
    plt.show()


def plot_and_save_direction_line(index, direction, init_positions, real_positions, guess_positions):
    indices = np.arange(n)

    plt.figure(figsize=(10, 5))
    plt.plot(indices, init_positions[index] - real_positions[index], color=init_color, label='Initial')
    plt.plot(indices, guess_positions[index] - real_positions[index], color=guess_color, label='Corrected (Guess)')

    plt.xlabel('Point Index')
    plt.ylabel(f'Average Score')
    plt.title(f'{direction}-Direction Position Comparison')
    plt.legend()

    # Save the figure
    plt.savefig(f'../model_load/{configs.model_name}/{direction}_direction_line.png')

    # Show the figure
    plt.show()


if __name__ == '__main__':
    # 创建config_manager，指定模型的配置文件

    base_dir = "../data/train"
    model = "DeepSet_Snapshot"
    model_config_name = f"{model}.yaml"
    model_args_path = f"../model_load/{model}/model.pt"
    configs = ConfigManager()
    configs.set_default_config(model_config_name)
    guess_positions, real_positions, init_positions = Predict(base_dir, model_args_path)
    init_color = 'green'
    guess_color = 'blue'
    real_color = 'red'
    # 计算初始位置和预测位置的分数
    init_scores = calc_score(init_positions, real_positions)
    guess_scores = calc_score(guess_positions, real_positions)
    time_points = range(len(init_scores))
    n = init_positions.shape[1]
    # 创建线图
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, init_scores, label='Initial Scores', color=init_color, marker='o')
    plt.plot(time_points, guess_scores, label='Predicted Scores', color=guess_color, marker='x')

    # 添加标题和标签
    plt.title('Comparison of Initial and Predicted Scores Over Time')
    plt.xlabel('Time Point')
    plt.ylabel('Score')

    # 添加图例
    plt.legend()

    # 保存线图到文件
    plt.savefig(f'../model_load/{configs.model_name}/score_lines.png')

    # 显示线图
    plt.show()

    # 创建柱状图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Initial Average Score', 'Predicted Average Score'],
                   [init_scores.mean(), guess_scores.mean()],
                   color=[init_color, guess_color])

    # 添加标题和标签
    plt.title('Comparison of Average Scores')
    plt.xlabel('Score Type')
    plt.ylabel('Average Score')

    # 在右上角添加柱状图的大小标注
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
                 ha='center', va='bottom')

    # 保存柱状图到文件
    plt.savefig(f'../model_load/{configs.model_name}/score_mean.png')

    # 显示柱状图
    plt.show()
    directions = ['X', 'Y', 'Z']
    # Plot and save each direction separately
    for i, direction in enumerate(directions):
        plot_and_save_direction_bar(i, direction, init_positions, real_positions, guess_positions)
        plot_and_save_direction_line(i, direction, init_positions, real_positions, guess_positions)
