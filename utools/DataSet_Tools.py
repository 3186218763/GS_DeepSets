import numpy as np
import pandas as pd
import pymap3d as pm
import scipy.optimize as optimize
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.auto import tqdm

from utools.Mylog import logger
from utools.Satellite_Tools import carrier_smoothing, satellite_selection, pr_residuals, \
    prr_residuals, jac_prr_residuals, jac_pr_residuals, los_vector

load_dotenv()


def normalize_residuals(residuals):
    """
    把样本的residuals归一化[-1, 1]
    :param residuals(一列一列的residual)
    :return: 归一化后的residuals
    """
    if residuals.shape[0] == 1:
        # 如果只有一行，直接返回全零数组
        return np.zeros_like(residuals)
    elif np.all(residuals == residuals[0]):
        # 如果所有行都一样，直接返回全零数组
        return np.zeros_like(residuals)
    else:
        min_data = np.min(residuals, axis=0)
        max_data = np.max(residuals, axis=0)

        normalized_data = 2 * (residuals - min_data) / (max_data - min_data) - 1

        return normalized_data


def normalize_rng(length):
    """
    把一组样本的los_vector中的距离归一化到[-, 1]
    :param length: rng
    :return: 归一化后的rng
    """
    if len(length.shape) == 1:
        # 如果长度只有一行，直接返回0
        if len(length) == 1:
            return np.array([0.0])
        # 如果一行里面的数都一样，直接返回0
        elif np.all(length == length[0]):
            return np.array([0.0])
    # 获取长度数据的最小值和最大值
    min_length = np.min(length)
    max_length = np.max(length)

    # 如果最小值和最大值相等，说明长度全都一样，直接返回0
    if min_length == max_length:
        return np.zeros(length.shape)

    # 归一化数据
    normalized_length = 2 * (length - min_length) / (max_length - min_length) - 1

    return normalized_length


def get_all_key_values(gnss_path, has_imv_data=False, imv_data_path=None, min_satellites=4, need_normalize=False):
    """
    Get all key-value pairs from a csv file.
    :param min_satellites: 要求最少从几个卫星接受数据才能进行los_vectors等等的计算，默认为4
    :param gnss_path:device_gnss.csv的路径
    :param has_imv_data:是否使用device_imu.csv数据,默认False
    :param imv_data_path:device_imu.csv,默认None
    :return:
    """
    gnss_df = pd.read_csv(gnss_path, low_memory=False)

    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    gnss_df = carrier_smoothing(gnss_df)

    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    x0 = np.zeros(4)
    v0 = np.zeros(4)
    x_wls = []
    v_wls = []
    key_data = []
    utc_list = []

    # 我暂时没有想到这些device的观察数据改怎么用
    if has_imv_data:
        imv_df = pd.read_csv(imv_data_path)
        non_zero_value = imv_df.loc[imv_df['BiasX'] != 0.0, 'BiasX'].iloc[0]
        imv_df['BiasX'] = imv_df['BiasX'].replace(0.0, non_zero_value)
        non_zero_value = imv_df.loc[imv_df['BiasY'] != 0.0, 'BiasY'].iloc[0]
        imv_df['BiasY'] = imv_df['BiasY'].replace(0.0, non_zero_value)
        non_zero_value = imv_df.loc[imv_df['BiasZ'] != 0.0, 'BiasZ'].iloc[0]
        imv_df['BiasZ'] = imv_df['BiasZ'].replace(0.0, non_zero_value)
        for i, (t_utc, df) in enumerate(tqdm(imv_df.groupby('utcTimeMillis'), total=len(utcTimeMillis))):
            Re_X = df['MeasurementX'] + df['BiasXMicroT']
            Re_Y = df['MeasurementY'] + df['BiasYMicroT']
            Re_Z = df['MeasurementZ'] + df['BiasZMicroT']

    for i, (t_utc, df) in tqdm(enumerate(gnss_df.groupby('utcTimeMillis')), total=len(gnss_df.groupby('utcTimeMillis')),
                               desc="正在从gnss_df中提前样本的体征"):

        # 选择满足条件的卫星数据
        df_pr = satellite_selection(df, 'pr_smooth', min_satellites)
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond', min_satellites)
        if df_pr.empty or df_prr.empty:
            print(f"不满足练条件, 第{i}组, utc={t_utc}被移除")

        else:
            # 计算 pseudorandom/pseudorange rate（距离残差和速度残差）
            pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
                  df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
            prr = (df_prr['PseudorangeRateMetersPerSecond'] +
                   df_prr['SvClockDriftMetersPerSecond']).to_numpy()

            # 计算卫星位置，和los_vectors
            xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                             'SvPositionZEcefMeters']].to_numpy()
            xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                               'SvPositionZEcefMeters']].to_numpy()
            vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                           'SvVelocityZEcefMetersPerSecond']].to_numpy()

            #  peseudorange/pseudorange rate的权重矩阵
            Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
            Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

            # 计算的wls初始值
            if len(pr) >= min_satellites:
                # Normal WLS
                if np.all(x0 == 0):
                    opt = optimize.least_squares(
                        pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                    x0 = opt.x

                opt = optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
                if opt.status < 1 or opt.status == 2:
                    print(f'i = {i} position lsq status = {opt.status}')
                    print(f"不满足练条件, 第{i}组, utc={t_utc}被移除")
                    continue
                else:
                    x_wls.append(opt.x[:3])
                    x0 = opt.x

            if len(prr) >= min_satellites:
                if np.all(v0 == 0):
                    opt = optimize.least_squares(
                        prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                    v0 = opt.x

                opt = optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
                if opt.status < 1:
                    print(f'i = {i} velocity lsq status = {opt.status}')
                    print(f"不满足练条件, 第{i}组, utc={t_utc}被移除")
                    continue
                else:
                    v_wls.append(opt.x[:3])
                    v0 = opt.x
                residuals_prr = prr_residuals(v0, vsat, prr, x0, xsat_prr, Wv)
                residuals_pr = (pr_residuals(x0, xsat_pr, pr, Wx))

                combined_residuals = np.hstack((residuals_pr.reshape(-1, 1), residuals_prr.reshape(-1, 1)))
                u, rng = los_vector(x0[:3], xsat_prr)
                los_vecs = np.concatenate((u, rng.reshape(-1, 1)), axis=1)
                key_values = np.hstack((combined_residuals, los_vecs))
                key_data.append(key_values)
                utc_list.append(t_utc)

    return utc_list, x_wls, v_wls, key_data


def get_samples(gnss_path, gt_path, normalize, standardize, debug, keep_init_real):
    gt_df = pd.read_csv(gt_path, low_memory=False)
    gt_utc_list = gt_df['UnixTimeMillis'].tolist()
    utc_list, x_wls, v_wls, key_data = get_all_key_values(gnss_path)

    not_null_rows = gt_df[gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].notnull().all(axis=1)]
    not_null_list = not_null_rows['UnixTimeMillis'].tolist()
    data_utc = np.intersect1d(gt_utc_list, utc_list)
    need_utc = np.intersect1d(data_utc, not_null_list)
    gt_jwd = gt_df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
    data_len = len(need_utc)
    samples = []
    for idx in range(data_len):
        utc = need_utc[idx]
        i_index = utc_list.index(utc)
        g_index = gt_utc_list.index(utc)
        latitude, longitude, altitude = gt_jwd[g_index]
        real_position = pm.geodetic2ecef(latitude, longitude, altitude)
        init_position = x_wls[i_index]
        features = key_data[i_index]
        if normalize:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)
        if standardize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        right_correction = real_position - init_position
        if keep_init_real:
            sample = {
                'right_correction': torch.tensor(right_correction, dtype=torch.float32),
                'features': torch.tensor(features, dtype=torch.float32),
                'init_position': torch.tensor(init_position, dtype=torch.float32),
                'real_position': torch.tensor(real_position, dtype=torch.float32)
            }
        else:
            sample = {
                'right_correction': torch.tensor(right_correction, dtype=torch.float32),
                'features': torch.tensor(features, dtype=torch.float32)
            }
        if debug:
            logger.info(f"{idx}:{sample}")
        samples.append(sample)

    return data_len, samples


def check_dataset(loader):
    """

    :param loader:虽然是检测dataset，但是请传入dataloader
    :return: 没有问题就没有提示
    """
    for i, batch_sample in enumerate(loader):
        _batch_sample, pad_mask = batch_sample
        features = _batch_sample['features']
        right_correction = _batch_sample['right_correction']

        # 检查特征和标签的形状是否正确
        if features.shape[0] != right_correction.shape[0]:
            error_msg = f"第 {i} 个批次中特征和标签的形状不匹配！"
            print(error_msg)
            raise ValueError(error_msg)

        # 检查特征是否包含 NaN 值
        if torch.isnan(features).any():
            error_msg = f"第 {i} 个批次中特征中存在 NaN 值！"
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 检查标签是否包含 NaN 值
        if torch.isnan(right_correction).any():
            error_msg = f"第 {i} 个批次中标签中存在 NaN 值！"
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 检查是否有无限值
        if torch.isinf(features).any() or torch.isinf(right_correction).any():
            error_msg = f"第 {i} 个批次中存在无限值！"
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)






