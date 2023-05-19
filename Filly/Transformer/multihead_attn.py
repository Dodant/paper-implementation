# paper: Attention is all you need

import math
from typing import Optional, List

import torch
from torch import nn
from labml import tracker


class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, heads:int, d_k: int, bias: bool):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads  # 4
        self.d_k = d_k  # 128

    def forward(self, x:torch.Tensor):
        # x.shape = 100, 32, 512 # [seq_len, batch_size, d_model]
        head_shape = x.shape[:-1]  # [seq_len, batch_size, d_model] or [batch_size, d_model]
        # head_shape = [100, 32]
        x = self.linear(x)
        # x.shape = 100, 32, 512
        x = x.view(*head_shape, self.heads, self.d_k)
        # x.shape = 100, 32, 4, 128
        return x  # [seq_len, batch_size, heads, d_k] or [batch_size, heads, d_model]


class MultiHeadAttention(nn.Module):
    # heads is the number of heads
    # d_model is the number of features in the Q, K, V vectors
    def __init__(self, heads:int, d_model:int, dropout_prob:float=0.1, bias:bool=True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads

        # Q,K,V - Linear(in_features=512, out_features=512, bias=True)
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query:torch.Tensor, key:torch.Tensor):
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask:torch.Tensor, query_shape: List[int], key_shape:List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, *,
                query:torch.Tensor,
                key:torch.Tensor,
                value:torch.Tensor,
                mask:Optional[torch.Tensor]=None):

        # 100, 32, 512
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        query = self.query(query)  # [100, 32, 4, 128]
        key = self.key(key)  # [100, 32, 4, 128]
        value = self.value(value)  # [100, 32, 4, 128]

        scores = self.get_scores(query, key)  # [100, 100, 32, 4]
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(scores)
        tracker.debug('attn', attn)
        attn = self.dropout(attn)
        # attn - 100, 100, 32, 4 / value - 100, 32, 4, 128
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        # x - [100, 32, 4, 128]
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        # x - [100, 32, 512]
        x = self.output(x)
        # x - [100, 32, 512]
        return x


heads = 4
d_model = 512

msa = MultiHeadAttention(heads, d_model)
x = msa(query=torch.randn(100, 32, d_model), key=torch.randn(100, 32, d_model), value=torch.randn(100, 32, d_model))
print(x.shape)  # [100, 32, 512]
