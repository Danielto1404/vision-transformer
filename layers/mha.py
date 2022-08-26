import math
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            heads: int,
            qkv_dim: Optional[int] = None,
            dropout=0.0,
            device: Union[str, torch.device] = 'cpu'
    ):
        super(MultiHeadAttention, self).__init__()

        self.qkv_dim = embedding_dim if qkv_dim is None else qkv_dim
        self.head_dim = self.qkv_dim // heads

        assert self.head_dim * heads == self.qkv_dim, "`embedding_dim` must be divisible by number of `heads`"

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.device = device

        self.WQ = torch.empty((embedding_dim, self.qkv_dim), requires_grad=True).to(device)
        self.WK = torch.empty((embedding_dim, self.qkv_dim), requires_grad=True).to(device)
        self.WV = torch.empty((embedding_dim, self.qkv_dim), requires_grad=True).to(device)
        self.W0 = torch.empty((self.qkv_dim, embedding_dim), requires_grad=True).to(device)

        self.__setup__()

    def __setup__(self):
        xavier_uniform_(self.WQ)
        xavier_uniform_(self.WK)
        xavier_uniform_(self.WV)
        xavier_uniform_(self.W0)

    def forward(self, x):
        """
        Applies multi-head self-attention to given sequence
        """
        q = x @ self.WQ
        k = x @ self.WK
        v = x @ self.WV

        # batch x seq x dim => batch x dim x seq
        k = k.transpose(2, 1)

        attentions = torch.bmm(q, k) / math.sqrt(self.head_dim)
        attentions = nn.functional.softmax(attentions, dim=-1)

        if self.training and self.dropout > 0.0:
            attentions = nn.functional.dropout(attentions, p=self.dropout)

        z = torch.bmm(attentions, v)

        x = z @ self.W0

        return x
