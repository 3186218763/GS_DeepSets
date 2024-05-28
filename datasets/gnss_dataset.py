import pandas as pd
from torch.utils.data import Dataset
import os
import pymap3d as pm
from utools.DataSet_Tools import get_all_key_values, get_samples
import numpy as np
import torch
from utools.Mylog import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Gnss_Dataset(Dataset):
    def __init__(self, target_dir, transform=None, target_transform=None, normalize=False, standardize=False,
                 debug=False, keep_init_real=False):
        """
        Gnss数据建立dataset
        :param keep_init_real: 是否获取real_position和init_position
        :param target_dir: 输入数据的路径
        :param transform:
        :param target_transform:
        :param normalize: 是否归一化
        :param standardize: 是否正则化
        :param debug: 是否起debug模式
        """

        gt_path = os.path.join(target_dir, 'ground_truth.csv')
        gnss_path = os.path.join(target_dir, 'device_gnss.csv')

        self.data_len, self.samples = get_samples(gnss_path, gt_path, normalize, standardize, debug, keep_init_real)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.samples[idx]


# gnss_dataset核心算法实现
if __name__ == '__main__':
    gnss_path = "../data/train/2020-07-08-22-28-us-ca/pixel4/device_gnss.csv"
    gt_path = "../data/train/2020-07-08-22-28-us-ca/pixel4/ground_truth.csv"

    gt_df = pd.read_csv(gt_path)
    gt_utc_list = gt_df['UnixTimeMillis'].tolist()
    utc_list, x_wls, v_wls, key_data = get_all_key_values(gnss_path)
    print(len(utc_list))
    print(len(x_wls))
    not_null_rows = gt_df[gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].notnull().all(axis=1)]
    not_null_list = not_null_rows['UnixTimeMillis'].tolist()
    data_utc = np.intersect1d(gt_utc_list, utc_list)
    data_utc = np.intersect1d(data_utc, not_null_list)
    gt_jwd = gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
    print(len(data_utc))
    for idx in range(len(data_utc)):
        utc = data_utc[idx]
        i_index = utc_list.index(utc)
        g_index = gt_utc_list.index(utc)
        logger.debug(f"x_wls;{len(x_wls)}, key_data:{len(key_data)}")
        logger.debug(f"utc={utc}, i_index={i_index}, g_index={g_index}")
        latitude, longitude, altitude = gt_jwd[g_index]
        real_position = pm.geodetic2ecef(latitude, longitude, altitude)
        init_position = x_wls[i_index]
        features = key_data[i_index]
