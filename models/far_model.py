"""FAR (Focused Action Refinement) for Frozen 1-bit VLA.

Architecture:
    Frozen BitVLA -> H0 image + Htop action tokens + a_base
                  -> FARCascadedEncoder (L layers of triple cascaded attention)
                  -> TokenRefiner (z_action + top_action_tokens -> refined tokens)
                  -> Trainable action head
                  -> Dual-level residual anchoring (token +/-0.3 tanh, pose +/-0.1 tanh)

Key design:
  * Three independent attention branches (H_A self, H_AQ cross, H_V focus) each
    with its own softmax, fused by an MLP. Avoids the query-side shortcut that
    arises when modalities share a single softmax.
  * Patch-level top-k=256 focus + element-wise channel gate on A->V path.
  * Bounded residual: pose_final = a_base + 0.1 * tanh(a_refine - a_base).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focus_cascaded_block import FocusCascadedBlock, FocusCascadedConfig


@dataclass
class FARConfig:
    # Keep this string compatible with the project-root eval_dsar_rollout.py
    # which dispatches FAR through its "dsar_v7_focus" code path.
    model_family: str = "dsar_v7_focus"

    model_dim: int = 2560
    qformer_dim: int = 640
    num_heads: int = 8
    num_cascaded_layers: int = 4

    chunk_size: int = 8
    action_dim: int = 7
    num_queries: int = 8

    topk_patches: int = 256
    use_channel_gate: bool = True
    use_patch_focus: bool = True

    # Input-side (token) anchor is relaxed vs v6 so the encoder can express
    # richer corrections; output-side (pose) anchor at 0.10 still caps rollout
    # deviation. Reviewer feedback pointed out the double 0.1 was too tight.
    token_delta_scale_init: float = 0.30
    pose_residual_scale: float = 0.10

    # Change 1: remove a_base conditioning from AQ (learned_query + chunk_embed only)
    aq_use_a_base: bool = False

    lr_encoder: float = 2e-4
    lr_head: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    grip_loss_weight: float = 1.0


class FARCascadedEncoder(nn.Module):
    """Triple-cascaded encoder.

    Inputs:
      img_tokens_h0      (B, 512, 2560)
      img_valid_h0       (B, 512)
      top_action_tokens_flat  (B, 56, 2560)
      a_base             (B, 8, 7)
    Outputs:
      z_action           (B, 8, 7, R)
      diagnostics from last layer
    """

    def __init__(self, cfg: FARConfig):
        super().__init__()
        self.cfg = cfg

        self.image_proj = nn.Linear(cfg.model_dim, cfg.qformer_dim)
        self.image_ln = nn.LayerNorm(cfg.qformer_dim)

        self.action_proj = nn.Linear(cfg.model_dim, cfg.qformer_dim)
        self.action_ln = nn.LayerNorm(cfg.qformer_dim)

        self.chunk_embed = nn.Embedding(cfg.chunk_size, cfg.qformer_dim)
        self.dim_embed = nn.Embedding(cfg.action_dim, cfg.qformer_dim)

        # Learned chunk queries (AQ)
        self.query_embed = nn.Parameter(
            torch.randn(cfg.num_queries, cfg.qformer_dim) * 0.02
        )
        if cfg.aq_use_a_base:
            self.base_chunk_proj = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.qformer_dim),
                nn.GELU(),
                nn.Linear(cfg.qformer_dim, cfg.qformer_dim),
            )
        self.query_ln = nn.LayerNorm(cfg.qformer_dim)

        block_cfg = FocusCascadedConfig(
            dim=cfg.qformer_dim,
            num_heads=cfg.num_heads,
            topk_patches=cfg.topk_patches,
            use_channel_gate=cfg.use_channel_gate,
            use_patch_focus=cfg.use_patch_focus,
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
        )
        self.blocks = nn.ModuleList([
            FocusCascadedBlock(block_cfg)
            for _ in range(cfg.num_cascaded_layers)
        ])

    def build_AQ(self, a_base: torch.Tensor) -> torch.Tensor:
        bsz, k, _ = a_base.shape
        device = a_base.device
        q = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        chunk_ids = torch.arange(k, device=device).view(1, k).expand(bsz, k)
        q = q + self.chunk_embed(chunk_ids)
        if self.cfg.aq_use_a_base:
            q = q + self.base_chunk_proj(a_base)
        return self.query_ln(q)

    def build_A(self, top_action_tokens_flat: torch.Tensor) -> torch.Tensor:
        bsz = top_action_tokens_flat.shape[0]
        device = top_action_tokens_flat.device
        cs, ad = self.cfg.chunk_size, self.cfg.action_dim

        A = self.action_proj(top_action_tokens_flat)

        chunk_ids = torch.arange(cs, device=device).view(cs, 1).expand(cs, ad).reshape(-1)
        dim_ids = torch.arange(ad, device=device).view(1, ad).expand(cs, ad).reshape(-1)

        A = A + self.chunk_embed(chunk_ids.view(1, -1).expand(bsz, -1))
        A = A + self.dim_embed(dim_ids.view(1, -1).expand(bsz, -1))
        return self.action_ln(A)

    def build_V(self, img_tokens: torch.Tensor) -> torch.Tensor:
        return self.image_ln(self.image_proj(img_tokens))

    def forward(
        self,
        *,
        img_tokens_h0: torch.Tensor,
        img_valid_h0: torch.Tensor,
        top_action_tokens_flat: torch.Tensor,
        a_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        A = self.build_A(top_action_tokens_flat)  # (B, 56, R)
        AQ = self.build_AQ(a_base)                 # (B, 8, R)
        V = self.build_V(img_tokens_h0)            # (B, 512, R)

        last_out = None
        for block in self.blocks:
            out = block(A, AQ, V, V_mask=img_valid_h0)
            A = out["A"]
            last_out = out

        z_action = A.reshape(
            A.shape[0], self.cfg.chunk_size, self.cfg.action_dim, self.cfg.qformer_dim
        )

        return {
            "A_final": A,
            "z_action": z_action,
            "H_A_contrib": last_out["H_A_contrib"],
            "H_AQ_contrib": last_out["H_AQ_contrib"],
            "H_V_contrib": last_out["H_V_contrib"],
            "H_V_raw_contrib": last_out["H_V_raw_contrib"],
            "v_attn_max": last_out["v_attn_max"],
            "v_attn_entropy": last_out["v_attn_entropy"],
            "v_attn_entropy_abs": last_out["v_attn_entropy_abs"],
            "gate_mean": last_out["gate_mean"],
        }


class TokenRefiner(nn.Module):
    """Same shape as v4/v6 refiner."""

    def __init__(self, cfg: FARConfig):
        super().__init__()
        self.cfg = cfg
        self.z_proj = nn.Linear(cfg.qformer_dim, cfg.model_dim)
        self.top_proj = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * cfg.model_dim),
            nn.Linear(2 * cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Linear(cfg.model_dim, cfg.model_dim),
        )
        self.delta_scale = nn.Parameter(torch.tensor(cfg.token_delta_scale_init))

    def forward(
        self,
        z_action: torch.Tensor,
        top_action_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = torch.cat(
            [self.z_proj(z_action), self.top_proj(top_action_tokens)], dim=-1
        )
        delta_h = torch.tanh(self.fuse(fused)) * self.delta_scale
        refined = top_action_tokens + delta_h
        return {
            "delta_h": delta_h,
            "refined_action_tokens": refined,
            "token_delta_norm": delta_h.norm(dim=-1),
        }


class FARModel(nn.Module):
    def __init__(self, cfg: FARConfig, full_trainable_action_head: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.encoder = FARCascadedEncoder(cfg)
        self.refiner = TokenRefiner(cfg)
        self.refined_action_head = full_trainable_action_head

    def forward(self, bridge: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc = self.encoder(
            img_tokens_h0=bridge["img_tokens_h0"],
            img_valid_h0=bridge["img_valid_h0"],
            top_action_tokens_flat=bridge["top_action_tokens_flat"],
            a_base=bridge["a_base"],
        )

        ref = self.refiner(
            z_action=enc["z_action"],
            top_action_tokens=bridge["top_action_tokens"],
        )

        bsz = bridge["a_base"].shape[0]
        refined_flat = ref["refined_action_tokens"].reshape(
            bsz, self.cfg.chunk_size * self.cfg.action_dim, self.cfg.model_dim
        )

        head_dtype = next(self.refined_action_head.parameters()).dtype
        a_refine_raw = self.refined_action_head.predict_action(
            refined_flat.to(dtype=head_dtype)
        ).to(torch.float32)

        a_base = bridge["a_base"]
        pose_base = a_base[..., :6]
        pose_refine = a_refine_raw[..., :6]
        pose_final = pose_base + self.cfg.pose_residual_scale * torch.tanh(
            pose_refine - pose_base
        )

        grip_logit = a_refine_raw[..., 6]
        grip_prob = torch.sigmoid(grip_logit)
        a_final = torch.cat([pose_final, grip_prob.unsqueeze(-1)], dim=-1)

        return {
            "a_base": a_base,
            "a_refine_raw": a_refine_raw,
            "a_final": a_final,
            "pose_final": pose_final,
            "grip_logit": grip_logit,
            "grip_prob": grip_prob,
            **enc,
            **ref,
        }


def far_loss(
    outputs: Dict[str, torch.Tensor],
    actions_gt: torch.Tensor,
    cfg: FARConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pose_gt = actions_gt[..., :6]
    grip_gt = (actions_gt[..., 6] > 0.0).to(actions_gt.dtype)

    l_pose = F.smooth_l1_loss(outputs["pose_final"], pose_gt, beta=0.02)
    l_grip = F.binary_cross_entropy_with_logits(outputs["grip_logit"], grip_gt)
    loss = l_pose + cfg.grip_loss_weight * l_grip

    grip_pred = (outputs["grip_prob"] > 0.5).to(grip_gt.dtype)
    grip_acc = (grip_pred == grip_gt).to(torch.float32).mean()

    a_base = outputs["a_base"]
    base_pose_mae = (a_base[..., :6] - pose_gt).abs().mean()
    final_pose_mae = (outputs["pose_final"] - pose_gt).abs().mean()
    improve_pct = 100.0 * (base_pose_mae - final_pose_mae) / base_pose_mae.clamp_min(1e-8)

    logs = {
        "loss": float(loss.detach().cpu()),
        "l_pose": float(l_pose.detach().cpu()),
        "l_grip": float(l_grip.detach().cpu()),
        "base_pose_mae": float(base_pose_mae.detach().cpu()),
        "final_pose_mae": float(final_pose_mae.detach().cpu()),
        "final_vs_base_improve_pct": float(improve_pct.detach().cpu()),
        "grip_acc": float(grip_acc.detach().cpu()),
        "token_delta_norm": float(outputs["token_delta_norm"].mean().detach().cpu()),
        "H_A_contrib": float(outputs["H_A_contrib"].detach().cpu()),
        "H_AQ_contrib": float(outputs["H_AQ_contrib"].detach().cpu()),
        "H_V_contrib": float(outputs["H_V_contrib"].detach().cpu()),
        "H_V_raw_contrib": float(outputs["H_V_raw_contrib"].detach().cpu()),
        "v_attn_max": float(outputs["v_attn_max"].detach().cpu()),
        "v_attn_entropy": float(outputs["v_attn_entropy"].detach().cpu()),
        "v_attn_entropy_abs": float(outputs["v_attn_entropy_abs"].detach().cpu()),
        "gate_mean": float(outputs["gate_mean"].detach().cpu()),
    }
    return loss, logs
