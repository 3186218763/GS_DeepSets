import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../data/base_station/csv/combined.csv', low_memory=False)

    print(df.head)
    # 打印列名
    print(df.columns)
