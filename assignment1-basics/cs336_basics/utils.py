import math
from typing import Iterable, Iterator, List, Dict, Tuple, Optional, Any
import torch
import numpy.typing as npt
from typing import IO, Any, BinaryIO
import os
import torch
import torch.nn as nn
from typing import Iterable, Iterator, List, Dict, Tuple, Optional, Any
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
import numpy as np

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if (it < warmup_iters):
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    return min_learning_rate + 1/2 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * ((it - warmup_iters) / (cosine_cycle_iters - warmup_iters))))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    对参数的梯度进行全局L2范数剪裁
    
    参数:
        parameters: 模型参数的可迭代对象
        max_l2_norm: 最大允许的L2范数阈值
    """
    # 收集所有存在的梯度
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads:  # 没有梯度需要处理
        return
    
    # 计算所有梯度的总L2范数平方和
    total_norm_squared = sum(torch.sum(torch.pow(g, 2)) for g in grads)
    total_norm = torch.sqrt(total_norm_squared)
    
    # 如果总范数超过阈值，则进行剪裁
    if total_norm > max_l2_norm:
        # 计算缩放系数
        clip_coef = max_l2_norm / (total_norm + 1e-6)  # 加小epsilon避免除零
        
        # 对每个梯度应用缩放系数
        for g in grads:
            g.mul_(clip_coef)  # 原地修改梯度


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']


def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    从数据集中获取一批样本用于语言模型训练
    
    Args:
        dataset: 输入数据集，一维数组
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备（如"cpu"或"cuda"）
        
    Returns:
        x: 输入序列 (batch_size, context_length)
        y: 目标序列 (batch_size, context_length)，比x偏移一个位置
    """
    # print(len(dataset), batch_size, context_length)
    max_start = len(dataset) - context_length
    # print('dataset', len(dataset))
    # print(max_start)
    x = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size, context_length))
    # starts = np.random.choice(max_start, size=batch_size, replace=False)
    starts = np.random.randint(0, max_start, size=batch_size)
    # starts[0] = 0
    # starts[len(starts) - 1] = max_start
    for i, start in enumerate(starts):
        x[i] = dataset[start: start+context_length]
        y[i] = dataset[start+1:start+context_length+1]
    x_tensor = torch.tensor(x, device=device)
    y_tensor = torch.tensor(y, device=device)

    return x_tensor.long(), y_tensor.long()


def cross_entropy(
    inputs: Float[Tensor, " batch_size ... vocab_size"], targets: Int[Tensor, " batch_size ..."]
) -> Float[Tensor, ""]:
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    batch_indices = torch.arange(inputs.size(0), device=inputs.device)  
    target_probs = inputs[batch_indices, targets]
    exp_in_features = torch.exp(inputs)
    res = torch.log(exp_in_features.sum(dim=-1, keepdim=True)) - target_probs
    return res.mean()