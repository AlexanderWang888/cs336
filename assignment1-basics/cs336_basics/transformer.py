import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.t()