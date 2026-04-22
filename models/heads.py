from __future__ import annotations

from pathlib import Path
import torch

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DDP-wrapped checkpoint keys."""
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _load_action_head_state_dict(checkpoint_dir: str | Path):
    ckpt_dir = Path(checkpoint_dir)
    return _strip_module_prefix(
        torch.load(ckpt_dir / "action_head--100000_checkpoint.pt", map_location="cpu")
    )


def _load_proprio_projector_state_dict(checkpoint_dir: str | Path):
    ckpt_dir = Path(checkpoint_dir)
    return _strip_module_prefix(
        torch.load(ckpt_dir / "proprio_projector--100000_checkpoint.pt", map_location="cpu")
    )


def build_frozen_heads(checkpoint_dir: str | Path, device: torch.device):
    """Load the official frozen coarse action head + proprio projector.

    This exactly matches the actual BitVLA codebase settings:
      - input_dim = hidden_dim = 2560
      - action_dim = 7
      - proprio_dim = 8 (LIBERO)
    """
    action_head = L1RegressionActionHead(
        input_dim=2560,
        hidden_dim=2560,
        action_dim=ACTION_DIM,
    )
    action_head.load_state_dict(_load_action_head_state_dict(checkpoint_dir))
    action_head.eval().to(device)
    for p in action_head.parameters():
        p.requires_grad = False

    proprio_projector = ProprioProjector(llm_dim=2560, proprio_dim=PROPRIO_DIM)
    proprio_projector.load_state_dict(_load_proprio_projector_state_dict(checkpoint_dir))
    proprio_projector.eval().to(device)
    for p in proprio_projector.parameters():
        p.requires_grad = False

    return action_head, proprio_projector


def build_trainable_action_head(checkpoint_dir: str | Path, device: torch.device):
    """Load the official BitVLA action head architecture with trainable weights."""
    action_head = L1RegressionActionHead(
        input_dim=2560,
        hidden_dim=2560,
        action_dim=ACTION_DIM,
    )
    action_head.load_state_dict(_load_action_head_state_dict(checkpoint_dir))
    action_head.train().to(device)
    for p in action_head.parameters():
        p.requires_grad = True
    return action_head


def build_dual_action_heads(checkpoint_dir: str | Path, device: torch.device):
    """Return frozen base head + trainable refined head + frozen proprio projector."""
    frozen_base_action_head, proprio_projector = build_frozen_heads(checkpoint_dir, device)
    refined_trainable_action_head = build_trainable_action_head(checkpoint_dir, device)
    return frozen_base_action_head, refined_trainable_action_head, proprio_projector
