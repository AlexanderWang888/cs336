from collections.abc import Callable, Iterable 
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e1):
        if lr < 0:
            raise ValueError(f"无效的学习率: {lr}")
        defaults = {"lr": lr} 
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            loss = closure() 
        
        for group in self.param_groups:
            lr = group["lr"]  # 获取当前参数组的学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取与参数 p 关联的状态（如迭代次数）
                state = self.state[p] 
                # 从状态中读取迭代次数，默认初始值为 0
                t = state.get("t", 0) 
                # 获取损失对参数 p 的梯度
                grad = p.grad.data 
                # 原地更新参数张量
                p.data -= lr / math.sqrt(t + 1) * grad 
                # 更新迭代次数
                state["t"] = t + 1 
        
        return loss 




class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e1, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5 ):
        if lr < 0:
            raise ValueError(f"无效的学习率: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "epsilon": eps, "lamb": weight_decay} 
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None 
        if closure is not None:
            loss = closure() 
        
        for group in self.param_groups:
            lr = group["lr"]  # 获取当前参数组的学习率
            beta1 = group['beta1']
            beta2 = group['beta2']
            lamb = group['lamb']
            epsilon = group['epsilon']
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取与参数 p 关联的状态（如迭代次数）
                grad = p.grad.data 
                data = p.data
                state = self.state[p] 
                # 从状态中读取迭代次数，默认初始值为 0
                m = state.get('m', torch.zeros_like(data))
                v = state.get('v', torch.zeros_like(data))
                t = state.get("t", 1) 
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                alpha = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 获取损失对参数 p 的梯度
                p.data -= alpha * m / (torch.sqrt(v) + epsilon)
                p.data -= lr * lamb * p.data
                # 更新迭代次数
                state["t"] = t + 1 
                state['m'] = m
                state['v'] = v
        
        return loss  


    # 初始化参数（示例：10×10 的随机张量）
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# # 初始化优化器（学习率设为 1）
# opt = SGD([weights], lr=1e2)

# for t in range(10):
#     # 重置所有可学习参数的梯度
#     opt.zero_grad() 
#     # 计算标量损失（示例：参数的平均平方损失）
#     loss = (weights**2).mean() 
#     print(loss.cpu().item())
#     # 反向传播：计算梯度
#     loss.backward() 
#     # 执行优化器步骤：更新参数
#     opt.step() 