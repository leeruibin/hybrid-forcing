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
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    # Q/OS/LSE 按 query 长度 L_q 索引; K/V 按 key 长度 L_k 索引
    q_offset = idx_bh * L_q * D
    kv_offset = idx_bh * L_k * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L_q
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + kv_offset + offs_n[None, :] * D + offs_d[:, None]
    V_ptrs = V + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * D + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_q)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L_k - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        if L_k - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L_q)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L_q)


@triton.jit
def _attn_fwd_bsld(
    Q, K, V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT, LSE, OS,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """(B, L, H, D) 布局: stride_L = H*D, LSE 为 (B, L_q, H)."""
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    stride_L = H * D
    base_q = (idx_bh // H) * L_q * H * D + (idx_bh % H) * D
    base_kv = (idx_bh // H) * L_k * H * D + (idx_bh % H) * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_base = (idx_bh // H) * L_q * H + (idx_bh % H)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    K_ptrs = K + base_kv + offs_n[None, :] * stride_L + offs_d[:, None]
    V_ptrs = V + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    OS_ptrs = OS + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    LSE_ptrs = LSE + lse_base + offs_m * H

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_q)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L_k - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * stride_L, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        if L_k - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs + idx_n * BLOCK_N * stride_L, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L_q)

    m_i += tl.math.log2(l_i)
    tl.store(LSE_ptrs, m_i, mask=offs_m < L_q)


@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,
    L_q: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    OS += idx_bh * L_q * D
    DOS += idx_bh * L_q * D
    DELTAS += idx_bh * L_q

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L_q)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L_q)

    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L_q)


@triton.jit
def _attn_bwd_preprocess_bsld(
    OS, DOS, DELTAS,
    L_q: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    stride_L = H * D
    base = (idx_bh // H) * L_q * H * D + (idx_bh % H) * D
    lse_base = (idx_bh // H) * L_q * H + (idx_bh % H)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    o_s = tl.load(OS + base + offs_m[:, None] * stride_L + offs_d[None, :], mask=offs_m[:, None] < L_q)
    do_s = tl.load(DOS + base + offs_m[:, None] * stride_L + offs_d[None, :], mask=offs_m[:, None] < L_q)

    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + lse_base + offs_m * H, delta_s, mask=offs_m < L_q)


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_offset = idx_bh * L_q * D
    kv_offset = idx_bh * L_k * D
    lse_offset = idx_bh * L_q
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    Q_ptrs = Q + q_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    DQ_ptrs = DQ + q_offset + offs_m[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    LUT_ptr = LUT + lut_offset

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_q)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L_q)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L_q)
    lse = tl.load(LSE_ptrs, mask=offs_m < L_q, other=float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L_k - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)

        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L_q)


@triton.jit
def _attn_bwd_dq_bsld(
    Q, K, V, LSE, DELTAS,
    DOS, DQ, LUT,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    stride_L = H * D
    base_q = (idx_bh // H) * L_q * H * D + (idx_bh % H) * D
    base_kv = (idx_bh // H) * L_k * H * D + (idx_bh % H) * D
    lse_base = (idx_bh // H) * L_q * H + (idx_bh % H)
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    K_ptrs = K + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    V_ptrs = V + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    DQ_ptrs = DQ + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    DOS_ptrs = DOS + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    LSE_ptrs = LSE + lse_base + offs_m * H
    DELTAS_ptrs = DELTAS + lse_base + offs_m * H
    LUT_ptr = LUT + lut_offset

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < L_q)
    do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L_q)
    delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L_q)
    lse = tl.load(LSE_ptrs, mask=offs_m < L_q, other=float("inf"))

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    for block_idx in tl.range(topk, num_stages=2):
        idx_n = tl.load(LUT_ptr + block_idx)
        n_mask = offs_n < L_k - idx_n * BLOCK_N

        k = tl.load(K_ptrs + idx_n * BLOCK_N * stride_L, mask=n_mask[:, None])
        v = tl.load(V_ptrs + idx_n * BLOCK_N * stride_L, mask=n_mask[:, None])
        qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
        p = tl.math.exp2(qk - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)

        dp = tl.dot(do_s, v.T).to(tl.float32)
        ds = p * (dp - delta_s[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L_q)


@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    q_offset = idx_bh * L_q * D
    kv_offset = idx_bh * L_k * D
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    lse_offset = idx_bh * L_q

    Q_ptrs = Q + q_offset + offs_m[:, None] * D + offs_d[None, :]
    K_ptrs = K + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    V_ptrs = V + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    DOS_ptrs = DOS + q_offset + offs_m[:, None] * D + offs_d[None, :]
    DK_ptrs = DK + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    DV_ptrs = DV + kv_offset + offs_n[:, None] * D + offs_d[None, :]
    LSE_ptrs = LSE + lse_offset + offs_m
    DELTAS_ptrs = DELTAS + lse_offset + offs_m
    KBID_ptr = KBID + kbid_offset + idx_n

    k = tl.load(K_ptrs, mask=offs_n[:, None] < L_k)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L_k)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L_q, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L_q - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)  # = 1 / ln(2)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L_k, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        Q_ptrs += BLOCK_M2 * D
        DOS_ptrs += BLOCK_M2 * D
        LSE_ptrs += BLOCK_M2
        DELTAS_ptrs += BLOCK_M2
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L_k)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L_k)


@triton.jit
def _attn_bwd_dkdv_bsld(
    Q, K, V, DOS, DK, DV,
    qk_scale, KBID, LSE, DELTAS,
    L_q: tl.constexpr,
    L_k: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    stride_L = H * D
    base_q = (idx_bh // H) * L_q * H * D + (idx_bh % H) * D
    base_kv = (idx_bh // H) * L_k * H * D + (idx_bh % H) * D
    lse_base = (idx_bh // H) * L_q * H + (idx_bh % H)
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    K_ptrs = K + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    V_ptrs = V + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    DOS_ptrs = DOS + base_q + offs_m[:, None] * stride_L + offs_d[None, :]
    DK_ptrs = DK + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    DV_ptrs = DV + base_kv + offs_n[:, None] * stride_L + offs_d[None, :]
    LSE_ptrs = LSE + lse_base + offs_m * H
    DELTAS_ptrs = DELTAS + lse_base + offs_m * H
    KBID_ptr = KBID + kbid_offset + idx_n

    k = tl.load(K_ptrs, mask=offs_n[:, None] < L_k)
    v = tl.load(V_ptrs, mask=offs_n[:, None] < L_k)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for idx_m in tl.range(0, L_q, BLOCK_M2):
        kbid = tl.load(KBID_ptr)
        if kbid == 1:
            m_mask = offs_m < L_q - idx_m
            q = tl.load(Q_ptrs, mask=m_mask[:, None])
            lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
            qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
            pT = tl.math.exp2(qkT - lse[None, :])
            pT = tl.where(offs_n[:, None] < L_k, pT, 0.0)

            do = tl.load(DOS_ptrs, mask=m_mask[:, None])
            dv += tl.dot(pT.to(do.dtype), do)
            delta = tl.load(DELTAS_ptrs, mask=m_mask)
            dpT = tl.dot(v, tl.trans(do))
            dsT = pT * (dpT - delta[None, :])
            dk += tl.dot(dsT.to(q.dtype), q)

        Q_ptrs += BLOCK_M2 * stride_L
        DOS_ptrs += BLOCK_M2 * stride_L
        LSE_ptrs += BLOCK_M2 * H
        DELTAS_ptrs += BLOCK_M2 * H
        if (idx_m + BLOCK_M2) % BLOCK_M == 0:
            KBID_ptr += N_BLOCKS

    tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L_k)
    tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L_k)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None, use_bsld=False):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()

        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        if use_bsld:
            B, L_q, H, D = q.shape
            L_k = k.shape[1]
        else:
            B, H, L_q, D = q.shape
            L_k = k.shape[2]
        assert (v.shape[1] if use_bsld else v.shape[2]) == L_k, "k and v must have the same sequence length"

        if qk_scale is None:
            qk_scale = D**-0.5

        M_BLOCKS = triton.cdiv(L_q, BLOCK_M)

        o_s = torch.empty_like(q)
        if use_bsld:
            lse = torch.empty((B, L_q, H), device=q.device, dtype=torch.float32)
        else:
            lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

        grid = (M_BLOCKS, B * H)
        if use_bsld:
            _attn_fwd_bsld[grid](
                q, k, v, qk_scale, topk,
                lut, lse, o_s,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS, H=H,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=4 if D == 64 else 8,
                num_stages=3
            )
        else:
            _attn_fwd[grid](
                q, k, v, qk_scale, topk,
                lut, lse, o_s,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=4 if q.shape[-1] == 64 else 8,
                num_stages=3
            )

        ctx.save_for_backward(q, k, v, k_block_id, lut, lse, o_s)
        ctx.qk_scale = qk_scale
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.use_bsld = use_bsld
        return o_s

    @staticmethod
    def backward(ctx, do_s):
        q, k, v, k_block_id, lut, lse, o_s = ctx.saved_tensors
        do_s = do_s.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        use_bsld = ctx.use_bsld
        if use_bsld:
            B, L_q, H, D = q.shape
            L_k = k.shape[1]
        else:
            B, H, L_q, D = q.shape
            L_k = k.shape[2]

        M_BLOCKS = triton.cdiv(L_q, BLOCK_M)
        N_BLOCKS = triton.cdiv(L_k, BLOCK_N)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta_s = torch.empty_like(lse)

        grid = (M_BLOCKS, B * H)
        if use_bsld:
            _attn_bwd_preprocess_bsld[grid](
                o_s, do_s, delta_s,
                L_q=L_q, H=H, D=D, BLOCK_M=BLOCK_M,
            )
            _attn_bwd_dq_bsld[grid](
                q, k, v, lse, delta_s,
                do_s, dq, lut,
                ctx.qk_scale, ctx.topk,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS, H=H,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=4 if D == 64 else 8,
                num_stages=4 if D == 64 else 5
            )
            grid = (N_BLOCKS, B * H)
            _attn_bwd_dkdv_bsld[grid](
                q, k, v, do_s, dk, dv,
                ctx.qk_scale, k_block_id, lse, delta_s,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS, N_BLOCKS=N_BLOCKS, H=H,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                BLOCK_SLICE_FACTOR=BLOCK_M // 64,
                num_warps=4 if D == 64 else 8,
                num_stages=4 if D == 64 else 5
            )
        else:
            _attn_bwd_preprocess[grid](
                o_s, do_s, delta_s,
                L_q=L_q, D=D, BLOCK_M=BLOCK_M,
            )
            _attn_bwd_dq[grid](
                q, k, v, lse, delta_s,
                do_s, dq, lut,
                ctx.qk_scale, ctx.topk,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=4 if q.shape[-1] == 64 else 8,
                num_stages=4 if q.shape[-1] == 64 else 5
            )
            grid = (N_BLOCKS, B * H)
            _attn_bwd_dkdv[grid](
                q, k, v, do_s, dk, dv,
                ctx.qk_scale, k_block_id, lse, delta_s,
                L_q=L_q, L_k=L_k, M_BLOCKS=M_BLOCKS, N_BLOCKS=N_BLOCKS,
                D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                BLOCK_SLICE_FACTOR=BLOCK_M // 64,
                num_warps=4 if q.shape[-1] == 64 else 8,
                num_stages=4 if q.shape[-1] == 64 else 5
            )

        return dq, dk, dv, None, None, None, None, None, None, None