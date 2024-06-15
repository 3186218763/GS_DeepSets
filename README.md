# gnss定位矫正复合神经网络
## 项目建议
本项目的测试环境的ubuntu 22.04, 显卡为Tesla p40, 本项目的建议显存至少11GB, 运行内存至少16GB
。本项目使用cuda11.8和12.2, 使用的torch为12.1

下面是本项目需要使用的库
```requirements
numpy~=1.26.4
pandas~=2.2.2
pymap3d~=3.1.0
matplotlib~=3.8.4
scipy~=1.13.0
tqdm~=4.66.2
scipy
torch
scikit-learn~=1.4.2
python-dotenv~=1.0.1
georinex
geopy~=2.4.1
joblib~=1.4.2
PyYAML~=6.0.1
Jinja2~=3.1.3
```

## 项目结构
```angular2html
GS_DeepSets
|  configs
|  convert_tool
|  data
   |  test
   |  train
      |  2020-07-08-22-28-us-ca
      |  ...
|  datasets
|  logs
|  nets
   |  DeepSet.py
   |  DeepSet_Snapshot.py
   |  ...
|  script
   |  predict.py
   |  train_model.py
|  utools
```

## 数据来源
我们的数据来源与Google Smartphone Decimeter Challenge 2023-2024。如果你想要下载数据，
[点击这里](https://www.kaggle.com/competitions/smartphone-decimeter-2023/data) 进入Kaggle下载，下载后根据代码结构轻微调整放入项目即可。 如果需要已经训练好的参数, 这里提供了DeepSet_Snapshot）[点击这里](https://pan.baidu.com/s/1dkGGK6HpZ7TarFwwlJTkeA?pwd=rsn5)
提取码：rsn5

## 注意事项
想要达到最佳效果需要慢慢调整参数，项目只有限条件的测试了部分地区的效果