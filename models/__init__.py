"""FAR (Focused Action Refinement) model package."""
from .far_model import (
    FARConfig,
    FARModel,
    FARCascadedEncoder,
    TokenRefiner,
    far_loss,
)
from .focus_cascaded_block import (
    FocusAttention,
    FocusCascadedBlock,
    FocusCascadedConfig,
    build_action_causal_mask,
)
from .bridge import (
    extract_far_features,
    build_far_inference_bridge,
)
from .heads import (
    build_frozen_heads,
    build_trainable_action_head,
    build_dual_action_heads,
)

__all__ = [
    "FARConfig", "FARModel", "FARCascadedEncoder", "TokenRefiner", "far_loss",
    "FocusAttention", "FocusCascadedBlock", "FocusCascadedConfig", "build_action_causal_mask",
    "extract_far_features", "build_far_inference_bridge",
    "build_frozen_heads", "build_trainable_action_head", "build_dual_action_heads",
]
