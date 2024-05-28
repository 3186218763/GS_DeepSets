import os
import tarfile
import shutil
import gzip
import subprocess
import csv
import urllib.request
from tqdm import tqdm
import pandas as pd
import georinex as gr

from datetime import datetime
from http.cookiejar import CookieJar
from concurrent.futures import ThreadPoolExecutor, as_completed

from geopy.distance import geodesic
from joblib import Parallel, delayed
from scipy.spatial import cKDTree


def download_data(year, day_of_year, hour, station, folder_path="../data/base_station/meta"):
    """
    使用本函数前，在root路径的.env文件配置IGS的用户信息

    :param year:
    :param day_of_year:
    :param hour:
    :param station:基站的名称
    :param folder_path:保存基站文件的路径，默认在../data/base_station/meta
    :return:
    """
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    print(username)
    print(password)
    output_gz = f'{station}{year}{day_of_year:03d}.crx.tar'
    file_path = os.path.join(folder_path, output_gz)
    # 检查文件是否已经存在
    if os.path.exists(file_path):
        return file_path

    url = f'https://cddis.nasa.gov/archive/gnss/data/highrate/{year}/{day_of_year:03d}/{station}_S_{year}{day_of_year:03d}0000_01D_01S_MO.crx.tar'
    print(url)
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

    cookie_jar = CookieJar()

    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(password_manager),
        urllib.request.HTTPCookieProcessor(cookie_jar))
    urllib.request.install_opener(opener)

    try:
        data_request = urllib.request.Request(url)
        data_request.add_header('Cookie', str(cookie_jar))
        data_response = urllib.request.urlopen(data_request)

        data_redirect_url = data_response.geturl()
        data_redirect_url += '&app_type=401'

        data_request = urllib.request.Request(data_redirect_url)
        data_response = urllib.request.urlopen(data_request)

        data_body = data_response.read()

        file_path = os.path.join(folder_path, output_gz)
        with open(file_path, 'wb') as file:
            file.write(data_body)

        # print(f"Success, {output_gz}, has downloaded to {folder_path}")

    except Exception as e:
        print("An error occurred:", e)


def calculate_distance(lat1, lon1, lat2, lon2):
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    return geodesic(point1, point2).kilometers


def convert_utc(timestamp_ms):
    # 将毫秒时间戳转换为秒，并转换为 datetime 对象
    return datetime.utcfromtimestamp(timestamp_ms / 1000.0)


def find_nearby_sites(row, igs_df, tree, range_km):
    target_lat = row['LatitudeDegrees']
    target_lon = row['LongitudeDegrees']
    current_utc = row['UnixTimeMillis']

    # 将 UTC 时间戳转换为 datetime 对象
    utc_datetime = convert_utc(current_utc)

    # 使用 KD-Tree 查询在范围内的站点索引
    indices = tree.query_ball_point((target_lat, target_lon), range_km / 111)  # 1 degree ~ 111 km

    results = []
    for idx in indices:
        site = igs_df.iloc[idx]
        result = {
            'StationName': site['#StationName'],  # 假设站点名称字段为 'StationName'
            'UTC': utc_datetime
        }
        results.append(result)
    return results


def find_near_station(gt_dir, range_km, igs_net_path="../data/base_station/IGSNetwork.csv"):
    """
    找到附近的基站
    :param gt_dir: 这里有个坑，为了方便，我直接使用gt_df，如果想使用gnss_df, 请处理里面重复的utc
    :param range_km: 设置多少km的范围
    :param igs_net_path: IGSNetwork.csv的表，如果没有请到官方网站下载
    :return: 返回的是DataFrame对象
    """
    gt_df = pd.read_csv(gt_dir, low_memory=False)
    igs_df = pd.read_csv(igs_net_path, low_memory=False)

    # 排除经纬度为空的行
    gt_df = gt_df.dropna(subset=['LatitudeDegrees', 'LongitudeDegrees'])

    # 构建KD-Tree以加速空间查询
    tree = cKDTree(igs_df[['Latitude', 'Longitude']].values)

    # 使用并行处理
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(find_nearby_sites)(row, igs_df, tree, range_km) for _, row in gt_df.iterrows()
    )

    # 将结果展平成一个列表
    results_flat = [item for sublist in results for item in sublist]

    result_df = pd.DataFrame(results_flat)
    return result_df


def download_all_data(result_df):
    """
    请在使用find_near_station后在使用这个函数
    :param result_df:
    :return:
    """
    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Downloading station data"):
        station = row['StationName']
        utc_datetime = row['UTC']

        # 转换日期和时间信息
        year = utc_datetime.year
        day_of_year = utc_datetime.day  # 将日期转换为一年中的第几天
        hour = utc_datetime.hour  # 提取小时
        # 下载数据
        download_data(year, day_of_year, hour, station)

    print("下载完成")


def untar_base_station_data(base_station_data_dir="../data/base_station/meta", untar_dir="../data/base_station/rnx"):
    for meta_name in os.listdir(base_station_data_dir):
        meta_path = os.path.join(base_station_data_dir, meta_name)
        try:
            with tarfile.open(meta_path, "r") as tar:
                tar.extractall(path=untar_dir)
        except Exception as e:
            print(f"An error occurred while extracting {untar_dir}: {e}")


def decompress_gz(file_path):
    """解压 .gz 文件"""
    output_file = file_path.replace('.gz', '')
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_file


def convert_crx_to_rnx(crx2rnx_exe_path, file_path):
    """使用 CRX2RNX 工具将 .crx 转换为 .rnx"""
    command = [crx2rnx_exe_path, file_path]
    subprocess.run(command, check=True)
    output_file = file_path.replace('.crx', '.rnx')
    return output_file


def process_crx_gz(crx2rnx_exe_path, file_path):
    """处理 .gz 文件"""
    try:

        # 解压 .gz 文件
        decompressed_gz = decompress_gz(file_path)

        # 删除原始 .gz 文件
        os.remove(file_path)

        # 使用 CRX2RNX 工具将 .crx 转换为 .rnx
        decompressed_rnx = convert_crx_to_rnx(crx2rnx_exe_path, decompressed_gz)

        # 删除原始 .crx 文件
        os.remove(decompressed_gz)

        return decompressed_rnx
    except Exception as e:
        msg = f'Failed to process {file_path}: {e}'
        raise Exception(msg)


def rinex_to_df(rinex_filename):
    """Convert RINEX file to pandas DataFrame"""
    obs_data = gr.load(rinex_filename)
    columns_of_interest = ['time', 'sv', 'C1X', 'C5X', 'C1C', 'C1W', 'C2W', 'C2X', 'C1P', 'C2C', 'C2P', 'C5I',
                           'L1X', 'L5X', 'L1C', 'L1W', 'L2W', 'L2X', 'L1P', 'L2C', 'L2P', 'L5I']
    obs_data = obs_data[columns_of_interest]
    df = obs_data.to_dataframe()

    return df


def process_folder(folder_path):
    """Process all .rnx files in a folder and return combined DataFrame"""
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith('.rnx'):
            rinex_file = os.path.join(folder_path, file)
            df = rinex_to_df(rinex_file)
            all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs)
        return combined_df
    else:
        msg = f"process_folder发生"
        raise Exception(msg)


def process_all_folders(base_folder_path, output_csv):
    """Process all folders and combine data into a single CSV"""
    all_dfs = []
    with ThreadPoolExecutor() as executor:
        futures = []

        for hour in range(24):
            folder_path = os.path.join(base_folder_path, f'{hour:02d}')
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                futures.append(executor.submit(process_folder, folder_path))
            else:
                print(f'Folder does not exist: {folder_path}')

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    if all_dfs:
        combined_df = pd.concat(all_dfs)
        combined_df.reset_index().to_csv(output_csv, index=False)
        print(f'Successfully created CSV: {output_csv}')
    else:
        print('No data processed.')


if __name__ == "__main__":
    input_folder = '../test/rnx/gnss/data/highrate/2020/025/20d'
    output_csv = '../data/base_station/csv/combined.csv'
    process_all_folders(input_folder, output_csv)


