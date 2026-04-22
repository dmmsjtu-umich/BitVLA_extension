"""Focus-Cascaded Block: triple cascaded attention + patch focus + channel gate.

Based on FocusVLA (Zhang et al. 2026), adapted for frozen 1-bit VLA refinement.

Key difference from v6 CEQFormer:
- Three INDEPENDENT attention modules (H_A, H_AQ, H_V) — A is never in the
  same softmax as AQ or V, so action queries cannot shortcut visual features.
- Patch-level top-k on A→V attention scores
- Element-wise channel gate on H_V output (not scalar like VLA-Adapter)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FocusCascadedConfig:
    dim: int = 640
    num_heads: int = 8
    topk_patches: int = 256
    ffn_mult: int = 4
    use_channel_gate: bool = True
    use_patch_focus: bool = True
    chunk_size: int = 8
    action_dim: int = 7


def build_action_causal_mask(
    n_action_tokens: int,
    chunk_size: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool causal mask over 56 action tokens by chunk index.

    Token at (chunk=k, dim=d) attends token (k', d') iff k' <= k.
    All action_dim at the same chunk step can attend each other.

    Returns a BOOL tensor where True = allowed. The block converts to an
    additive mask at forward time using `torch.finfo(dtype).min`, which
    survives bf16/fp16 (float('-inf') would become a finite large value in
    bf16 and leak probability to masked positions after softmax).
    """
    total = chunk_size * action_dim
    chunk_idx = torch.arange(total, device=device) // action_dim  # (56,)
    allowed = chunk_idx.unsqueeze(1) >= chunk_idx.unsqueeze(0)    # bool (56, 56)
    return allowed


class FocusAttention(nn.Module):
    """A -> V cross-attention with patch-level top-k and element-wise gate."""

    def __init__(self, dim: int, num_heads: int, topk: int, use_gate: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.topk = topk
        self.use_gate = use_gate

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.q_ln = nn.LayerNorm(dim)
        self.v_ln = nn.LayerNorm(dim)

        if use_gate:
            self.gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            # Open the gate at init: bias=+2 -> sigmoid(2) ≈ 0.88, so H_V
            # starts contributing strongly rather than being halved by a
            # sigmoid(0)=0.5 default.
            nn.init.constant_(self.gate[1].bias, 2.0)

    def forward(
        self,
        A: torch.Tensor,
        V: torch.Tensor,
        V_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, Na, D = A.shape
        _, Nv, _ = V.shape
        H = self.num_heads
        Dh = self.head_dim

        A_norm = self.q_ln(A)
        V_norm = self.v_ln(V)

        Q = self.q_proj(A_norm).view(B, Na, H, Dh).transpose(1, 2)
        K = self.k_proj(V_norm).view(B, Nv, H, Dh).transpose(1, 2)
        Val = self.v_proj(V_norm).view(B, Nv, H, Dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, Na, Nv)

        if V_mask is not None:
            scores = scores.masked_fill(~V_mask[:, None, None, :], float("-inf"))

        # Patch-level Focus: per-(batch, head, action) independent top-k
        # (matches FocusVLA Eq. 9, not the earlier loose "max over queries")
        if self.topk < Nv:
            _, topk_idx = scores.topk(self.topk, dim=-1)           # (B, H, Na, K)
            topk_mask_full = torch.zeros_like(scores, dtype=torch.bool)
            topk_mask_full.scatter_(dim=-1, index=topk_idx, value=True)
            scores = scores.masked_fill(~topk_mask_full, float("-inf"))
            n_valid_keys = float(self.topk)
        else:
            n_valid_keys = float(
                V_mask.sum(dim=-1).float().mean().item()
                if V_mask is not None else Nv
            )

        attn = F.softmax(scores, dim=-1)
        attn_avg = attn.mean(dim=1)  # (B, Na, Nv)

        H_V = attn @ Val  # (B, H, Na, Dh)
        H_V = H_V.transpose(1, 2).contiguous().view(B, Na, D)
        H_V = self.o_proj(H_V)

        # Save the pre-gate vision signal so we can report how strong the
        # attention itself is, independent of how much the gate lets through.
        # Reviewer's point: gate ∈ [0,1] systematically suppresses H_V.norm()
        # by ~12-47%, so H_V_contrib (post-gate) under-reports visual usage.
        H_V_raw = H_V

        if self.use_gate:
            g = self.gate(A)
            H_V = H_V * g
            gate_mean = g.mean()
        else:
            gate_mean = torch.tensor(1.0, device=A.device, dtype=A.dtype)

        # Diagnostics
        # attn_max: mean over batch/action of the peak attention weight
        # (computed on head-averaged attention; robust to multi-head spread).
        attn_max = attn_avg.amax(dim=-1).mean()

        # Entropy computed per-head (each head's softmax has support ≤ topk)
        # so ent_abs ∈ [0, log(topk)] and ent_norm ∈ [0, 1] are well-defined.
        # Averaging heads BEFORE entropy would inflate the support up to Nv
        # when different heads pick different patches.
        n_valid = max(float(n_valid_keys), 2.0)
        n_valid_log = math.log(n_valid)
        p = attn.clamp_min(1e-8)                       # (B, H, Na, Nv)
        ent_per_head_q = -(p * p.log()).sum(dim=-1)    # (B, H, Na)
        ent_norm = (ent_per_head_q / n_valid_log).mean()   # relative to log(topk)
        ent_abs = ent_per_head_q.mean()                     # absolute, nats

        return {
            "H_V": H_V,
            "H_V_raw": H_V_raw,
            "attn_max": attn_max,
            "attn_entropy": ent_norm,
            "attn_entropy_abs": ent_abs,
            "gate_mean": gate_mean,
            "attn_avg": attn_avg,          # (B, Na, Nv) head-averaged — for depth diag
        }


class FocusCascadedBlock(nn.Module):
    """One block of Focus-Cascaded Encoder.

    Three independent attentions, fused via MLP (not concatenated into one softmax):
        H_A  = CausalSelfAttn(A, A)        chunk-causal within action tokens
        H_AQ = CrossAttn(A, AQ)            action -> chunk queries
        H_V  = FocusAttn(A, V)             action -> vision (top-k + gate)
        A    = A + FusionMLP([H_A, H_AQ, H_V]) + FFN(...)
    """

    def __init__(self, cfg: FocusCascadedConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim
        H = cfg.num_heads

        self.self_attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.aq_attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.v_attn = FocusAttention(D, H, cfg.topk_patches, use_gate=cfg.use_channel_gate)

        self.ln_self_q = nn.LayerNorm(D)
        self.ln_aq_q = nn.LayerNorm(D)
        self.ln_aq_k = nn.LayerNorm(D)

        # Per-branch LN before fusion: keeps each branch at comparable scale
        # and prevents FusionMLP from collapsing to a single-branch shortcut.
        self.ln_fuse_A = nn.LayerNorm(D)
        self.ln_fuse_AQ = nn.LayerNorm(D)
        self.ln_fuse_V = nn.LayerNorm(D)

        self.fusion = nn.Sequential(
            nn.Linear(3 * D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, cfg.ffn_mult * D),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * D, D),
        )

        # Bool buffer; converted to an additive mask at forward time using the
        # active dtype's finfo.min so it stays correct under bf16/fp16.
        mask_bool = build_action_causal_mask(
            cfg.chunk_size * cfg.action_dim,
            cfg.chunk_size,
            cfg.action_dim,
            torch.device("cpu"),
        )
        self.register_buffer("causal_mask_bool", mask_bool, persistent=False)

    def forward(
        self,
        A: torch.Tensor,
        AQ: torch.Tensor,
        V: torch.Tensor,
        V_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1) H_A: causal self-attention over action tokens
        # Materialize the additive causal mask in A's dtype using finfo.min so
        # it survives bf16/fp16 (plain float('-inf') would be clamped).
        neg_inf = torch.finfo(A.dtype).min
        mask_add = torch.zeros(
            self.causal_mask_bool.shape, device=A.device, dtype=A.dtype,
        )
        mask_add.masked_fill_(
            ~self.causal_mask_bool.to(A.device), neg_inf,
        )

        A_norm = self.ln_self_q(A)
        H_A, _ = self.self_attn(
            query=A_norm, key=A_norm, value=A_norm,
            attn_mask=mask_add,
            need_weights=False,
        )

        # 2) H_AQ: action -> chunk queries
        Q_aq = self.ln_aq_q(A)
        K_aq = self.ln_aq_k(AQ)
        H_AQ, _ = self.aq_attn(
            query=Q_aq, key=K_aq, value=K_aq,
            need_weights=False,
        )

        # 3) H_V: action -> vision (focus)
        v_out = self.v_attn(A, V, V_mask=V_mask)
        H_V = v_out["H_V"]                 # post-gate
        H_V_raw = v_out["H_V_raw"]         # pre-gate

        # 4) Fusion + FFN (per-branch LN keeps the 3 paths at comparable scale)
        fused = self.fusion(torch.cat([
            self.ln_fuse_A(H_A),
            self.ln_fuse_AQ(H_AQ),
            self.ln_fuse_V(H_V),
        ], dim=-1))
        A = A + fused
        A = A + self.ffn(A)

        # Diagnostics: per-path norm contribution
        # H_V_contrib      : how much visual signal actually reaches the fusion
        #                    (post-gate). This is what fusion sees.
        # H_V_raw_contrib  : how strong the attention itself is, absent the
        #                    gate scaling. Better diagnostic for "is the
        #                    focus-attn learning to use vision".
        norm_A = H_A.norm(dim=-1).mean()
        norm_AQ = H_AQ.norm(dim=-1).mean()
        norm_V = H_V.norm(dim=-1).mean()
        norm_V_raw = H_V_raw.norm(dim=-1).mean()
        total = norm_A + norm_AQ + norm_V + 1e-8
        total_raw = norm_A + norm_AQ + norm_V_raw + 1e-8

        return {
            "A": A,
            "H_A_norm": norm_A,
            "H_AQ_norm": norm_AQ,
            "H_V_norm": norm_V,
            "H_V_raw_norm": norm_V_raw,
            "H_A_contrib": norm_A / total,
            "H_AQ_contrib": norm_AQ / total,
            "H_V_contrib": norm_V / total,
            "H_V_raw_contrib": norm_V_raw / total_raw,
            "v_attn_max": v_out["attn_max"],
            "v_attn_entropy": v_out["attn_entropy"],
            "v_attn_entropy_abs": v_out["attn_entropy_abs"],
            "gate_mean": v_out["gate_mean"],
        }
