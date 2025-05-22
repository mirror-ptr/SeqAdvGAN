import torch
import torch.nn as nn
from typing import Callable, Any # Import Callable and Any for type hinting

class Lambda(nn.Module):
    """
    一个简单的 PyTorch 层，它将任意函数作为其 forward 操作。
    允许在 nn.Sequential 或其他 nn.Module 容器中使用简单的函数。
    """
    def __init__(self, func: Callable[[Any], Any]): # Type hint for func: Callable with Any input/output
        """
        初始化 Lambda 层。

        Args:
            func (Callable[[Any], Any]): 要应用的函数。这个函数应该接受一个张量作为输入并返回一个张量。
        """
        super(Lambda, self).__init__()
        self.func: Callable[[Any], Any] = func # Store the function, add type hint

    def forward(self, x: torch.Tensor) -> torch.Tensor: # Type hint input and output
        """
        在输入张量上应用存储的函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 函数应用于输入张量后的结果张量。
        """
        return self.func(x)