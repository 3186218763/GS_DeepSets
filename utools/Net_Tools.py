import torch.nn as nn
import torch
from utools.Mylog import logger


def gradient_hook(module, grad_input, grad_output):
    # 这里可以访问梯度信息，进行监视或修改
    logger.debug("Gradient of module:", module)
    logger.debug("Input gradient:", grad_input)
    logger.debug("Output gradient:", grad_output)


class Integrated_Net(nn.Module):
    def __init__(self, *models: nn.Module):
        super(Integrated_Net, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        out = x
        for model in self.models:
            out = model(out)
        return out


def check_phi_permutation_invariance(phi_function, input_size=6):
    """
    phi函数必须满足置换不变
    """
    batch_size, num_groups = 64, 32
    input_tensor = torch.randn(batch_size, num_groups, input_size)
    permuted_indices = torch.randperm(num_groups)
    input_tensor_permuted = input_tensor[:, permuted_indices, :]

    output_original = phi_function(input_tensor)
    output_permuted = phi_function(input_tensor_permuted)

    # 检查输出是否相同或非常接近
    if torch.allclose(output_original, output_permuted, atol=1e-5):
        print(f"{phi_function.__class__.__name__} 满足！")
    else:
        print(f"{phi_function.__class__.__name__} 不满足！")


