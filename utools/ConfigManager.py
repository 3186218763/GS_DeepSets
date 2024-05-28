import os
import yaml
import numpy as np

import os
import yaml
import numpy as np


import os
import yaml
import numpy as np

class ConfigManager:
    def __init__(self, config_dir="../configs"):
        """
        读取configs的工具,能读取不同格式的config
        :param config_dir: 输入configs文件夹路径
        """
        self.config_dir = config_dir
        self.config_name = None
        self.configs = self.load_configs()
        self._update_attributes()

    def load_configs(self):
        configs = {}
        for filename in os.listdir(self.config_dir):
            config_name, extension = os.path.splitext(filename)
            if extension == ".yaml":
                with open(os.path.join(self.config_dir, filename), "r") as file:
                    configs[config_name] = yaml.safe_load(file)
        return configs

    def set_default_config(self, config_name):
        """
        :param config_name: 请输入配置的名称，请带后缀.yaml
        """
        self.config_name = config_name
        self._update_attributes()

    def _update_attributes(self):
        if self.config_name is not None:
            config_values = self.configs.get(os.path.splitext(self.config_name)[0], {})
            for key, value in config_values.items():
                # 尝试转换配置值的类型
                value = self._convert_value(value)
                setattr(self, key, value)

    def _convert_value(self, value):
        # 如果值为字符串，尝试转换成数字类型或 None
        if isinstance(value, str):
            value = value.strip()
            if value.lower() == "none":
                return None
            try:
                # 尝试转换为整数
                value = int(value)
            except ValueError:
                try:
                    # 尝试转换为浮点数
                    value = float(value)
                except ValueError:
                    try:
                        # 尝试转换为布尔值
                        if value.lower() == "true":
                            return True
                        elif value.lower() == "false":
                            return False
                    except ValueError:
                        pass  # 如果无法转换，则保持原样

        # 如果值为列表，尝试转换成 numpy 数组
        if isinstance(value, list):
            try:
                value = np.array(value)
            except ValueError:
                pass  # 如果无法转换，则保持原样

        return value

    def get_config(self, key):
        if self.config_name is None:
            raise ValueError("具体config文件路径还没有设置，请使用set_default_config进行设置")
        return getattr(self, key, None)




# 使用示例
if __name__ == "__main__":
    configs = ConfigManager()
    configs.set_default_config("DeepSet_Only.yaml")  # 设置默认的配置名称，带后缀
    # 有黄色虚心是正常的，放心使用
    print(type(configs.num_workers))
    print(type(configs.pad_columns))
    print(type(configs.Debug))
    print(type(configs.eps))
    print(configs.eps)
    print(type(configs.pad_columns))

