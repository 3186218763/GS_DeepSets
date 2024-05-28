import torch.nn as nn
import torch
from utools.Mylog import logger


def gradient_hook(module, grad_input, grad_output):
    # 这里可以访问梯度信息，进行监视或修改
    logger.debug("Gradient of module:", module)
    logger.debug("Input gradient:", grad_input)
    logger.debug("Output gradient:", grad_output)


class Integrated_Net(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module):
        super(Integrated_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out_model1 = self.model1(x)
        out = self.model2(out_model1)
        return out
