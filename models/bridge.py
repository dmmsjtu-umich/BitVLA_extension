"""FAR feature bridge — extract H0 image + Htop action tokens + a_base from frozen BitVLA.

Consolidates what used to be a 3-level wrapper chain (bridge_v7_focus ->
bridge_qformer_ceq_v6 -> bridge_qformer_action_refine_head_v4) into a single
module with `extract_far_features` (training-time) and
`build_far_inference_bridge` (closed-loop / rollout-time).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

from .far_model import FARConfig


_IMAGE_TOKEN_ID = 128010
_PROPRIO_TOKEN_ID = 128011
_STOP_TOKEN_ID = 128001
_PAD_TOKEN_ID = 128002
_BOS_TOKEN_ID = 128000


# ----- Helpers -----

def _batched_gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, _, dim = x.shape
    gather_idx = indices.unsqueeze(-1).expand(bsz, indices.shape[1], dim)
    return torch.gather(x, dim=1, index=gather_idx)


def _action_mask_to_indices(action_mask: torch.Tensor, chunk_size: int, action_dim: int) -> torch.Tensor:
    """Convert a dense action-position mask (B, T) into ordered indices (B, K, A)."""
    bsz = action_mask.shape[0]
    total = chunk_size * action_dim
    out: List[torch.Tensor] = []
    for b in range(bsz):
        idx = torch.nonzero(action_mask[b], as_tuple=False).squeeze(-1)
        if idx.numel() != total:
            raise ValueError(f"Expected {total} action positions, got {idx.numel()} for sample {b}")
        out.append(idx)
    return torch.stack(out, dim=0).view(bsz, chunk_size, action_dim)


def _gather_action_tokens(x: torch.Tensor, action_token_indices: torch.Tensor) -> torch.Tensor:
    """x: (B, T, D), indices: (B, K, A) -> (B, K, A, D)"""
    bsz, _, dim = x.shape
    k, a = action_token_indices.shape[1], action_token_indices.shape[2]
    flat_idx = action_token_indices.reshape(bsz, k * a)
    gathered = _batched_gather_tokens(x, flat_idx)
    return gathered.reshape(bsz, k, a, dim)


def _pad_masked_tokens(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length masked token subsets into padded tensors."""
    bsz, _, dim = x.shape
    counts = mask.sum(dim=1)
    nmax = max(int(counts.max().item()), 1)
    device = x.device
    padded = torch.zeros(bsz, nmax, dim, device=device, dtype=x.dtype)
    valid = torch.zeros(bsz, nmax, device=device, dtype=torch.bool)
    for b in range(bsz):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n > 0:
            padded[b, :n] = x[b, idx]
            valid[b, :n] = True
    return padded, valid


def _build_instruction_mask_aligned(
    batch: Dict[str, torch.Tensor],
    action_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Infer text-token mask from aligned input sequence, excluding image / proprio / action / special tokens."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    aligned_ids = input_ids[:, :-1]
    aligned_attn = attention_mask[:, :-1]

    action_pos_mask = torch.zeros_like(aligned_attn)
    flat_idx = action_token_indices.reshape(action_token_indices.shape[0], -1)
    action_pos_mask.scatter_(1, flat_idx, True)

    if "instruction_mask" in batch:
        instr_mask = batch["instruction_mask"][:, :-1].to(aligned_attn.device).bool()
        return instr_mask & aligned_attn & ~action_pos_mask

    proprio_mask = aligned_attn & (aligned_ids == _PROPRIO_TOKEN_ID) & ~action_pos_mask
    image_mask = aligned_attn & (aligned_ids == _IMAGE_TOKEN_ID)
    special_mask = (
        (aligned_ids == _PAD_TOKEN_ID)
        | (aligned_ids == _BOS_TOKEN_ID)
        | (aligned_ids == _STOP_TOKEN_ID)
    )
    return aligned_attn & ~image_mask & ~proprio_mask & ~action_pos_mask & ~special_mask


# ----- Training-time bridge -----

@torch.no_grad()
def extract_far_features(
    base_model,
    batch: Dict[str, torch.Tensor],
    frozen_base_action_head,
    proprio_projector,
    cfg: FARConfig,
) -> Dict[str, torch.Tensor]:
    """Training-time feature extraction.

    Returns a dict with:
      img_tokens_h0       (B, Nimg, model_dim)   — layer-0 image tokens (padded)
      img_valid_h0        (B, Nimg)               — validity mask
      text_tokens_h0      (B, Ntxt, model_dim)
      text_valid          (B, Ntxt)
      top_action_tokens   (B, K, A, model_dim)    — layer-30 action tokens
      top_action_tokens_flat (B, K*A, model_dim)
      a_base              (B, K, A)               — frozen head's action prediction
    """
    device = next(base_model.parameters()).device
    base_dtype = next(base_model.parameters()).dtype

    proprio_input = batch.get("proprio", None)
    if proprio_input is not None and proprio_projector is not None:
        prop_dtype = next(proprio_projector.parameters()).dtype
        proprio_input = proprio_input.to(device=device, dtype=prop_dtype)

    output = base_model.forward(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        pixel_values=batch["pixel_values"].to(device=device, dtype=base_dtype),
        labels=batch["labels"].to(device),
        output_hidden_states=True,
        proprio=proprio_input,
        proprio_projector=proprio_projector,
    )

    labels_shifted = batch["labels"][:, 1:].to(device)
    current_mask = get_current_action_mask(
        labels_shifted, ignore_index=-100, action_token_begin_idx=_PROPRIO_TOKEN_ID,
    )
    next_mask = get_next_actions_mask(
        labels_shifted, ignore_index=-100, action_token_begin_idx=_PROPRIO_TOKEN_ID,
    )
    action_mask = current_mask | next_mask
    action_token_indices = _action_mask_to_indices(action_mask, cfg.chunk_size, cfg.action_dim)

    aligned_ids = batch["input_ids"][:, :-1].to(device)
    aligned_attn = batch["attention_mask"][:, :-1].to(device).bool()

    action_pos_mask = torch.zeros_like(aligned_attn)
    flat_idx = action_token_indices.reshape(action_token_indices.shape[0], -1)
    action_pos_mask.scatter_(1, flat_idx, True)

    image_mask = aligned_attn & (aligned_ids == _IMAGE_TOKEN_ID)
    instruction_mask = _build_instruction_mask_aligned(batch, action_token_indices).to(device)

    hidden_states_aligned = [hs[:, :-1, :].to(torch.float32) for hs in output.hidden_states]
    h0 = hidden_states_aligned[0]
    htop = hidden_states_aligned[30]

    img_tokens_h0, img_valid = _pad_masked_tokens(h0, image_mask)
    text_tokens_h0, text_valid = _pad_masked_tokens(h0, instruction_mask)

    top_action_tokens = _gather_action_tokens(htop, action_token_indices)
    top_action_tokens_flat = top_action_tokens.reshape(
        top_action_tokens.shape[0], cfg.chunk_size * cfg.action_dim, cfg.model_dim,
    )

    head_dtype = next(frozen_base_action_head.parameters()).dtype
    a_base = frozen_base_action_head.predict_action(
        top_action_tokens_flat.to(dtype=head_dtype)
    ).to(torch.float32)

    return {
        "img_tokens_h0": img_tokens_h0,
        "img_valid_h0": img_valid,
        "text_tokens_h0": text_tokens_h0,
        "text_valid": text_valid,
        "top_action_tokens": top_action_tokens,
        "top_action_tokens_flat": top_action_tokens_flat,
        "a_base": a_base,
    }


# ----- Inference-time (rollout) bridge -----

@torch.no_grad()
def build_far_inference_bridge(
    vla,
    inputs: Dict[str, torch.Tensor],
    proprio: Optional[torch.Tensor],
    frozen_base_action_head,
    proprio_projector,
    cfg: FARConfig,
) -> Dict[str, torch.Tensor]:
    """Inference-time feature extraction (used during LIBERO rollout)."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    labels = input_ids.clone()
    labels[:] = vla.ignore_idx
    input_ids_ext, attention_mask_ext = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels_ext = vla._prepare_labels_for_action_prediction(labels, input_ids_ext)
    input_embeddings = vla.get_input_embeddings()(input_ids_ext)
    all_actions_mask = vla._process_action_masks(labels_ext)

    if pixel_values is not None:
        vision_feature_layer = vla.config.vision_feature_layer
        vision_feature_select_strategy = vla.config.vision_feature_select_strategy
        _, num_images, channels, height, width = pixel_values.shape
        pv_flat = pixel_values.view(-1, channels, height, width)
        img_emb = vla.get_image_features(
            pixel_values=pv_flat,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        img_emb = img_emb.view(-1, img_emb.shape[-1])
        img_mask = (input_ids_ext == vla.image_token_idx).unsqueeze(-1).expand_as(input_embeddings)
        input_embeddings = input_embeddings.masked_scatter(
            img_mask, img_emb.to(input_embeddings.device, input_embeddings.dtype),
        )

    if proprio_projector is not None and proprio is not None:
        prop_dtype = next(proprio_projector.parameters()).dtype
        prop_t = torch.as_tensor(proprio, device=input_embeddings.device, dtype=prop_dtype).reshape(1, -1)
        prop_feat = proprio_projector(prop_t)
        prop_flat = prop_feat.unsqueeze(1).view(-1, prop_feat.shape[-1])
        prop_mask = (input_ids_ext == vla.proprio_pad_idx).unsqueeze(-1).expand_as(input_embeddings)
        input_embeddings = input_embeddings.masked_scatter(
            prop_mask, prop_flat.to(input_embeddings.device, input_embeddings.dtype),
        )

    input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)

    llava_output = LlavaForConditionalGeneration.forward(
        vla,
        input_ids=None,
        attention_mask=attention_mask_ext,
        position_ids=None,
        pixel_values=None,
        labels=None,
        inputs_embeds=input_embeddings,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        use_bi_attn=True,
    )

    hidden_states_aligned = [hs[:, :-1, :].to(torch.float32) for hs in llava_output.hidden_states]
    labels_shifted = labels_ext[:, 1:]
    action_mask = (labels_shifted != vla.ignore_idx) & (labels_shifted != vla.stop_index)
    action_token_indices = _action_mask_to_indices(action_mask, cfg.chunk_size, cfg.action_dim)

    aligned_batch = {"input_ids": input_ids_ext, "attention_mask": attention_mask_ext}
    aligned_ids = input_ids_ext[:, :-1]
    aligned_attn = attention_mask_ext[:, :-1].bool()
    action_pos_mask = torch.zeros_like(aligned_attn)
    flat_idx = action_token_indices.reshape(action_token_indices.shape[0], -1)
    action_pos_mask.scatter_(1, flat_idx, True)

    image_mask = aligned_attn & (aligned_ids == vla.image_token_idx)
    instruction_mask = _build_instruction_mask_aligned(aligned_batch, action_token_indices).to(aligned_ids.device)

    h0 = hidden_states_aligned[0]
    htop = hidden_states_aligned[30]

    img_tokens_h0, img_valid = _pad_masked_tokens(h0, image_mask)
    text_tokens_h0, text_valid = _pad_masked_tokens(h0, instruction_mask)

    top_action_tokens = _gather_action_tokens(htop, action_token_indices)
    top_action_tokens_flat = top_action_tokens.reshape(
        top_action_tokens.shape[0], cfg.chunk_size * cfg.action_dim, cfg.model_dim,
    )

    head_dtype = next(frozen_base_action_head.parameters()).dtype
    a_base = frozen_base_action_head.predict_action(
        top_action_tokens_flat.to(dtype=head_dtype)
    ).to(torch.float32)

    return {
        "img_tokens_h0": img_tokens_h0,
        "img_valid_h0": img_valid,
        "text_tokens_h0": text_tokens_h0,
        "text_valid": text_valid,
        "top_action_tokens": top_action_tokens,
        "top_action_tokens_flat": top_action_tokens_flat,
        "a_base": a_base,
    }
