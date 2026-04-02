""" 
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention}, 
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import _attention
from .utils import get_block_map, get_cuda_arch

class SparseAttention(nn.Module):
    def __init__(self, head_dim, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True, tie_feature_map_qk=True, use_linear=True):
        R'''
        Args:
            head_dim: dimension of each head.
            topk: ratio of keys selected for sparse attention, shared across all queries.
            feature_map: feature map for linear attention, one of ['hedgehog', 'elu', 'relu', 'softmax'].
            BLKQ: block size for query.
            BLKK: block size for key.
            use_bf16: whether to use bfloat16 (default) or float16 for computation. The conversion to bf16/fp16 is done inside the module.
            tie_feature_map_qk: whether to use the same feature map for query and key.
        '''
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK

        # if tie_feature_map_qk:
        #     self.feature_map_k = self.feature_map_q

    def forward(self, q, k, v, return_sparsity=False, layout="bhld"):
        R'''
        Args:
            q, k, v: layout='bhld' 时为 (B, H, L, D)；layout='bsld' 时为 (B, seq_len, n_head, n_dim)，不做 transpose。
            return_sparsity: whether to return the actual sparsity.
            layout: 'bhld' 或 'bsld'，与 to_q/to_k/to_v 输出一致时用 'bsld' 可省 transpose。
        '''
        use_bsld = layout == "bsld"
        if use_bsld:
            assert k.shape[1] == v.shape[1], "k and v must have the same sequence length"
        else:
            assert k.shape[2] == v.shape[2], "k and v must have the same sequence length"
        dtype = q.dtype

        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK, layout_bhld=not use_bsld
        )

        q = q.contiguous().to(self.dtype)
        k = k.contiguous().to(self.dtype)
        v = v.contiguous().to(self.dtype)
        o_s = _attention.apply(q, k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK, None, use_bsld)

        o = o_s

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        return o