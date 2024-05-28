import numpy as np
import pandas as pd
import pymap3d as pm
import pymap3d.vincenty as pmv
import scipy.optimize as optimize
from tqdm.auto import tqdm
from scipy.spatial import distance
from utools.Constants import OMGE, CLIGHT


# 根据卫星的载波频率误差、仰角和载噪比（C/N0）来选择符合条件的卫星。
def satellite_selection(df, column, n=4):
    idx = df[column].notnull()  # 选择非空的行
    idx &= df['CarrierErrorHz'] < 2.0e6  # 选择载波频率误差小于2.0e6 Hz（即 2 MHz） (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # 继续细化选择仰角大于 10 度的卫星 (deg)
    idx &= df['Cn0DbHz'] > 15.0  # 只选择载噪比（C/N0）大于 15.0 dB-Hz 的卫星 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0  # 多径效应指示器的值为1，表示有多径效应, 0 则没有多径效应

    return df[idx]


# 计算从用户到卫星的视线向量
def los_vector(xusr, xsat):
    """

    :param xusr: 用户的位置坐标
    :param xsat: 卫星的位置坐标
    :return: u: 归一化后的用户到卫星的向量; rng: 被重塑为一维数组的用户到卫星的距离
    """
    u = xsat - xusr  # 计算用户到卫星的向量 u
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)  # 计算了用户到卫星的距离
    u /= rng  # 归一化u

    return u, rng.reshape(-1)


# 伪距残差的雅可比矩阵
def jac_pr_residuals(x, xsat, pr, W):
    """

    :param x: 这个向量包含了用户的（经度、纬度、高度）以及钟差（时间偏差）。
    :param xsat: 卫星的位置坐标
    :param pr: 伪距观测值
    :param W: 权重矩阵
    :return: 加权雅可比矩阵
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


# 计算伪距残差
def pr_residuals(x, xsat, pr, W):
    """

    :param x: 这个向量包含了用户的（经度、纬度、高度）以及钟差（时间偏差）。
    :param xsat: 卫星的位置坐标
    :param pr: 伪距观测值
    :param W: 权重矩阵
    :return: 加权的伪距残差
    """
    u, rng = los_vector(x[:3], xsat)

    # 对伪距观测值进行了修正，考虑了地球自转的影响
    rng += OMGE * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / CLIGHT

    # 添加了GPS L1钟差的修正量
    residuals = rng - (pr - x[3])

    return residuals @ W


# 计算伪距速率残差的雅可比矩阵
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """

    :param v:
    :param vsat:
    :param prr: 伪距观测值
    :param x: 这个向量包含了用户的（经度、纬度、高度）以及钟差（时间偏差）。
    :param xsat: 卫星的位置坐标
    :param W: 权重矩阵
    :return: 得到了加权雅可比矩阵
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return np.dot(W, J)


# 计算伪距速率残差
def prr_residuals(v, vsat, prr, x, xsat, W):
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat - v[:3]) * u, axis=1) \
           + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                              - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W


# 对伪距观测值进行载波平滑
def carrier_smoothing(gnss_df):
    carr_th = 1.0  # carrier phase jump threshold [m] 2->1.5 (best)->1.0
    pr_th = 15.0  # pseudorange jump threshold [m] 20->15

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate((gnss_df.groupby(['Svid', 'SignalType']))):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2 ** 1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2 ** 2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th  # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th  # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
                gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth

    return gnss_df


# 使用Vincenty 公式计算两个地理位置之间的大圆距离
def vincenty_distance(llh1, llh2):
    """

    :param llh1: 一组地理位置坐标（经度、纬度）
    :param llh2: 另外一组地理位置坐标（经度、纬度）
    :return: 所有大圆距离的数组。
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# 计算定位精度的分数
def calc_score(llh, llh_gt):
    """

    :param llh: 预测的地面位置坐标
    :param llh_gt: 地面实际的地理位置坐标
    :return: 最后返回平均分数
    """

    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score


# GNSS单点定位算法
def point_positioning(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    residuals = np.full([nepoch, 2], np.nan)

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
                # Robust WLS for position estimation
            opt = optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
            else:
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls


# 简单的异常值检测和插值过程
def exclude_interpolate_outlier(x_wls, v_wls):
    # Up velocity threshold
    v_up_th = 2.0  # m/s

    # Coordinate conversion
    x_llh = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
    v_enu = np.array(pm.ecef2enuv(
        v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], x_llh[0, 0], x_llh[0, 1])).T

    # Up velocity jump detection
    # Cars don't jump suddenly!
    idx_v_out = np.abs(v_enu[:, 2]) > v_up_th
    v_wls[idx_v_out, :] = np.nan

    # Interpolate NaNs at beginning and end of array
    x_df = pd.DataFrame({'x': x_wls[:, 0], 'y': x_wls[:, 1], 'z': x_wls[:, 2]})
    x_df = x_df.interpolate(limit_area='outside', limit_direction='both')

    # Interpolate all NaN data
    v_df = pd.DataFrame({'x': v_wls[:, 0], 'y': v_wls[:, 1], 'z': v_wls[:, 2]})
    v_df = v_df.interpolate(limit_area='outside', limit_direction='both')
    v_df = v_df.interpolate('spline', order=3)

    return x_df.to_numpy(), v_df.to_numpy()
