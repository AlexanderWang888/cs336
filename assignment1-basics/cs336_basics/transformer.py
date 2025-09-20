import torch
import torch.nn as nn
from typing import Iterable, Iterator, List, Dict, Tuple, Optional, Any
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np
import math
from einops import einsum,reduce,rearrange

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device:None = None, dtype:None = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        std_dev = math.sqrt(2 / (self.d_in + self.d_out))
        self.w = nn.Parameter(torch.empty((d_out, d_in), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.w, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)

    def forward(self, x: Tensor) -> torch.Tensor:
        return x @ self.w.t()
    


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.table = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.table, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> torch.Tensor:
        result = self.table[token_ids]
        return result
    


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones((self.d_model), device=self.device, dtype=self.dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        res = x / rms * self.g
        res.to(in_dtype)
        return res
        

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=self.device, dtype=self.dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=self.device, dtype=self.dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=self.device, dtype=self.dtype))

        std_dev = math.sqrt(2 / (self.d_ff + self.d_model))
        
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = x @ self.w1.t()
        res2 = ((res1) / (1 + torch.exp(-res1))) * (x @ self.w3.t())
        res3 = res2 @ self.w2.t()
        return res3


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    # torch.exp(in_features)
    in_features = in_features - in_features.max(dim=dim, keepdim=True).values
    exp_in_features = torch.exp(in_features)
    return exp_in_features / exp_in_features.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,)  -> Float[Tensor, " ... queries d_v"]:
    res1 = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        res2 = res1.masked_fill(~mask, -1e9)
    else:
        res2 = res1
    res3 = softmax(res2, -1)
    return res3 @ V
    


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化RoPE
        Args:
            theta: 旋转角度基数
            d_k: 输入Q或K向量的维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算旋转复数矩阵
        self.register_buffer("rope", self._precompute_freqs_cis(), persistent=False)
    
    def _precompute_freqs_cis(self) -> torch.Tensor:
        """
        预计算频率和相位
        Returns:
            形状为(max_seq_len, d_k)的张量，包含旋转位置编码
        """
        # 计算\theta_i序列，也就是频率序列
        # theta_i = 1 / { theta^{2i / d_k} }
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device)[:(self.d_k // 2)] / self.d_k))
        # 生成序列索引m [0, 1, ..., max_seq_len-1]
        seq_idx = torch.arange(0, self.max_seq_len, device=self.device)
        # 计算 m * \theta_i 矩阵
        freqs = einsum(seq_idx, freqs, "seq, d -> seq d")

        # 复数化
        # freqs[m][i] = m * \theta_i
        # freqs_cis[m][i] = 1 * e^{i * m * \theta_i}
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，形状为(..., seq_len, d_k)
            token_positions: 位置索引，形状为(..., seq_len)
        Returns:
            旋转位置编码后的张量，形状为(..., seq_len, d_k)
        """
        # 将维度分组
        x_ = rearrange(x, "... seq (d two) -> ... seq d two", two=2).float()
        # 转为复数(... seq (d 2) )
        x_ = torch.view_as_complex(x_)

        # 根据token_positions获取对应的位置的频率
        rope_pos = self.rope[token_positions]  # (batch, ..., seq_len, d_k // 2)

        # 旋转，之后转回实数域并展平
        x_out = rearrange(torch.view_as_real(x_ * rope_pos), "... seq d two -> ... seq (d two)", two=2)
        
        return x_out.to(x.dtype)  # 转回原始dtype
    


class Multihead_Self_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len:int|None = None,rope_theta: float = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = nn.Parameter(torch.empty((self.d_k * self.num_heads, self.d_model), device=device, dtype=dtype))
        self.k_proj = nn.Parameter(torch.empty((self.d_k * self.num_heads, self.d_model), device=device, dtype=dtype))
        self.v_proj = nn.Parameter(torch.empty((self.d_k * self.num_heads, self.d_model), device=device, dtype=dtype))
        self.o_proj = nn.Parameter(torch.empty((self.d_model, self.d_k * self.num_heads), device=device, dtype=dtype))
        std_dev = math.sqrt(2 / (self.d_model + self.d_model))
        nn.init.trunc_normal_(self.q_proj, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        nn.init.trunc_normal_(self.k_proj, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        nn.init.trunc_normal_(self.v_proj, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        nn.init.trunc_normal_(self.o_proj, mean=0.0, std=std_dev, a=-3 * std_dev, b=3 * std_dev)
        
        self.register_buffer("mask", torch.triu(torch.ones(max_seq_len, max_seq_len, device=device), diagonal=1) == 0)
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = RotaryPositionalEmbedding(theta=10000.0, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, in_features,token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        batch, seq_len, d_model  = in_features.shape
        dim = d_model // self.num_heads
        q_ff = (in_features @ self.q_proj.transpose(-1, -2)).view(batch, seq_len, self.num_heads, dim).permute(0, 2, 1, 3)
        k_ff = (in_features @ self.k_proj.transpose(-1, -2)).view(batch, seq_len, self.num_heads, dim).permute(0, 2, 1, 3)
        v_ff = (in_features @ self.v_proj.transpose(-1, -2)).view(batch, seq_len, self.num_heads, dim).permute(0, 2, 1, 3)

        # token_positions = torch.arange(seq_len, device=in_features.device)  # (seq,)
        # token_positions = token_positions.unsqueeze(0).expand(batch, seq_len)  # (batch, seq)

        q_ff = self.rope(q_ff, token_positions)  # (batch, heads, seq, d_k)
        k_ff = self.rope(k_ff, token_positions)
        
        if self.mask is not None:
            res = scaled_dot_product_attention(q_ff, k_ff, v_ff, self.mask[:seq_len, :seq_len])
        else:
            res = scaled_dot_product_attention(q_ff, k_ff, v_ff, None)
        res2 = res.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return res2 @ self.o_proj.transpose(-1, -2)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len:int, rope_theta: float = None, device=None, dtype=None):
        super().__init__()
        self.ffn = SwiGLU(d_model, d_ff, device=device)
        self.mla = Multihead_Self_Attention(d_model, num_heads, max_seq_len, rope_theta, device=device)
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)

    def forward(self, in_features: Float[Tensor, "batch sequence_length d_model"]) -> Float[Tensor, "batch sequence_length d_model"]:
        token_positions = torch.arange(in_features.shape[-2], dtype=torch.int, device=in_features.device)  # (batch, ..., seq_len)
        y = in_features + self.mla(self.norm1(in_features),token_positions)
        z = y + self.ffn(self.norm2(y))
        return z
        


class Transformer(nn.Module):
    def __init__(self, vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float, device=None, dtype=None):
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        # 修正2：循环逻辑（range(num_layers) 生成层索引）
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            )
        self.embedding = Embedding(vocab_size, d_model, device=device)
        self.norm = RMSNorm(d_model, device=device)
        self.lin = Linear(d_model, vocab_size, device=device)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"])-> Float[Tensor, " batch_size sequence_length vocab_size"]:
        res = self.embedding(in_indices)
        for block in self.transformer_blocks:
            res = block(res)
        res = self.norm(res)
        res = self.lin(res)
        # res = softmax(res, -1)
        return res