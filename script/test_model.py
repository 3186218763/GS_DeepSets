from script.train_model import Train_Model


if __name__ == '__main__':
    base_dir = "../data/train"
    model_config_name = "DeepSet_Dense.yaml"
    Train_Model(base_dir, model_config_name)



