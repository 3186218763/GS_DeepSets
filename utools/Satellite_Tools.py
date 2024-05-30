import numpy as np

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





