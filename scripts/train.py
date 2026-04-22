"""Train FAR (Focused Action Refinement)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import tqdm
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]              # repo root
# (no longer needed — repo root is the reference)
BITVLA_ROOT = REPO_ROOT / "repos" / "BitVLA" / "openvla-oft"
sys.path.insert(0, str(REPO_ROOT))                       # so `import models` works
sys.path.insert(0, str(BITVLA_ROOT))
sys.path.insert(0, str(BITVLA_ROOT / "vla-scripts"))

from transformers import (  # noqa: E402
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    LlavaProcessor,
    SiglipImageProcessor,
)

from bitvla import (  # noqa: E402
    Bitnet_ActionTokenizer,
    BitVLA_RLDSBatchTransform,
    BitVLAForActionPrediction,
    Bitvla_Config,
    Bitvla_PaddedCollatorForActionPrediction,
)
from bitvla.constants import (  # noqa: E402
    BITNET_ACTION_TOKEN_BEGIN_IDX,
    BITNET_DEFAULT_IMAGE_TOKEN,
    BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    BITNET_DEFAULT_IM_END_TOKEN,
    BITNET_IGNORE_INDEX,
    BITNET_PROPRIO_PAD_IDX,
    BITNET_STOP_INDEX,
)
from models import build_dual_action_heads  # noqa: E402
from models import extract_far_features  # noqa: E402
from models import FARConfig, FARModel, far_loss
from prismatic.vla.datasets import RLDSDataset  # noqa: E402
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # noqa: E402


def emit_progress_line(text: str) -> None:
    if sys.stdout.isatty() and sys.stderr.isatty():
        tqdm.tqdm.write(text)
    else:
        print(text, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FAR (Focused Action Refinement)")

    p.add_argument("--base_checkpoint", type=str, required=True)
    p.add_argument("--data_root_dir", type=str, default="data/modified_libero_rlds")
    p.add_argument("--dataset_name", type=str, default="libero_10_no_noops")
    p.add_argument("--output_dir", type=str, default="outputs/far")

    p.add_argument("--qformer_dim", type=int, default=640)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_cascaded_layers", type=int, default=4)
    p.add_argument("--topk_patches", type=int, default=256)
    p.add_argument("--use_channel_gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_patch_focus", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--token_delta_scale_init", type=float, default=0.30)
    p.add_argument("--pose_residual_scale", type=float, default=0.10)
    p.add_argument("--grip_loss_weight", type=float, default=1.0)

    p.add_argument("--lr_encoder", type=float, default=2e-4)
    p.add_argument("--lr_head", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr_ratio", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accumulation_steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--shuffle_buffer_size", type=int, default=10000)
    p.add_argument("--image_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_wrist_image", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--probe_batches", type=int, default=20)
    p.add_argument("--probe_freq", type=int, default=500)
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument(
        "--resume_reset_scheduler",
        action="store_true",
        help="When resuming, DON'T load scheduler state. For extending "
             "training with new --max_steps, this avoids LR schedule jumps."
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> FARConfig:
    return FARConfig(
        qformer_dim=args.qformer_dim,
        num_heads=args.num_heads,
        num_cascaded_layers=args.num_cascaded_layers,
        topk_patches=args.topk_patches,
        use_channel_gate=args.use_channel_gate,
        use_patch_focus=args.use_patch_focus,
        token_delta_scale_init=args.token_delta_scale_init,
        pose_residual_scale=args.pose_residual_scale,
        grip_loss_weight=args.grip_loss_weight,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )


def register_bitvla_hf_components() -> None:
    try:
        AutoConfig.register("bitvla", Bitvla_Config)
    except ValueError:
        pass
    try:
        AutoImageProcessor.register(Bitvla_Config, SiglipImageProcessor)
    except ValueError:
        pass
    try:
        AutoProcessor.register(Bitvla_Config, LlavaProcessor)
    except ValueError:
        pass
    try:
        AutoModelForVision2Seq.register(Bitvla_Config, BitVLAForActionPrediction)
    except ValueError:
        pass


def load_frozen_bitvla(checkpoint_path: str, device: torch.device):
    register_bitvla_hf_components()
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    vla.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )
    for parameter in vla.parameters():
        parameter.requires_grad = False
    vla.eval()
    processor.tokenizer.pad_token_id = 128002
    return vla, processor


def make_batch_transform(processor, use_wrist_image: bool):
    action_tokenizer = Bitnet_ActionTokenizer(processor.tokenizer, bins=252)
    return BitVLA_RLDSBatchTransform(
        action_tokenizer,
        processor=processor,
        use_wrist_image=use_wrist_image,
        use_proprio=True,
        end_token=BITNET_DEFAULT_IM_END_TOKEN,
        ignore_token_idx=BITNET_IGNORE_INDEX,
        image_token=BITNET_DEFAULT_IMAGE_TOKEN,
    )


def make_dataloader(
    data_root_dir: str,
    dataset_name: str,
    processor,
    batch_size: int,
    shuffle_buffer_size: int,
    image_aug: bool,
    use_wrist_image: bool,
):
    batch_transform = make_batch_transform(processor, use_wrist_image=use_wrist_image)
    dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=shuffle_buffer_size,
        image_aug=image_aug,
    )
    collator = Bitvla_PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        ignore_idx=BITNET_IGNORE_INDEX,
        padding_side="right",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    return dataset, dataloader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def lr_lambda_builder(max_steps: int, warmup_steps: int, min_lr_ratio: float):
    def _fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(max_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _fn


@torch.no_grad()
def run_probe(
    model: FARModel,
    probe_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    cfg: FARConfig,
    base_model,
    frozen_base_action_head,
    proprio_projector,
):
    model.eval()
    totals: Dict[str, float] = {}
    n = 0

    for batch in probe_batches:
        batch_gpu = move_batch_to_device(batch, device)
        bridge = extract_far_features(
            base_model, batch_gpu, frozen_base_action_head, proprio_projector, cfg,
        )
        outputs = model(bridge)
        actions_gt = batch_gpu["actions"].to(torch.float32)
        _, logs = far_loss(outputs, actions_gt, cfg)
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n += 1

    model.train()
    return {f"probe/{k}": v / max(n, 1) for k, v in totals.items()}


def build_probe_cache(probe_loader: Iterable[Dict[str, torch.Tensor]], num_batches: int):
    cache = []
    for i, batch in enumerate(probe_loader):
        if i >= num_batches:
            break
        cache.append({k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
    return cache


def save_checkpoint(
    path: Path,
    model: FARModel,
    optimizer,
    scheduler,
    global_step: int,
    cfg: FARConfig,
):
    state = {
        "step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": asdict(cfg),
    }
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: FARModel,
    optimizer=None,
    scheduler=None,
    device=None,
) -> int:
    state = torch.load(path, map_location=device if device is not None else "cpu")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = build_config(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading frozen BitVLA...")
    vla, processor = load_frozen_bitvla(args.base_checkpoint, device)
    print("Loading dual action heads (frozen base + trainable refined)...")
    frozen_base_action_head, refined_action_head, proprio_projector = build_dual_action_heads(
        args.base_checkpoint, device,
    )

    print("Creating DSAR v7-Focus model...")
    model = FARModel(cfg, refined_action_head).to(device).to(torch.float32)
    model.train()

    with open(output_dir / "far_cfg.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_dataset, train_loader = make_dataloader(
        args.data_root_dir, args.dataset_name, processor,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
        use_wrist_image=args.use_wrist_image,
    )
    save_dataset_statistics(train_dataset.dataset_statistics, output_dir)

    _, probe_loader = make_dataloader(
        args.data_root_dir, args.dataset_name, processor,
        batch_size=args.batch_size,
        shuffle_buffer_size=min(args.shuffle_buffer_size, 1000),
        image_aug=False,
        use_wrist_image=args.use_wrist_image,
    )
    probe_cache = build_probe_cache(probe_loader, num_batches=args.probe_batches)

    encoder_params = list(model.encoder.parameters()) + list(model.refiner.parameters())
    head_params = list(model.refined_action_head.parameters())
    n_enc = sum(p.numel() for p in encoder_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"Trainable: encoder+refiner={n_enc:,}  head={n_head:,}  total={n_enc+n_head:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.lr_encoder},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_builder(args.max_steps, args.warmup_steps, args.min_lr_ratio),
    )

    global_step = 0
    if args.resume_from:
        scheduler_to_load = None if args.resume_reset_scheduler else scheduler
        global_step = load_checkpoint(
            args.resume_from, model, optimizer, scheduler_to_load, device,
        )
        if args.resume_reset_scheduler:
            scheduler.last_epoch = global_step
            print(
                f"[resume_reset_scheduler] scheduler state NOT loaded; "
                f"last_epoch set to {global_step} so new max_steps={args.max_steps} "
                f"cosine governs LR from here."
            )
        print(f"Resumed from step={global_step}")

    train_log_file = open(output_dir / "train_log.csv", "w", newline="")
    probe_log_file = open(output_dir / "probe_log.csv", "w", newline="")
    train_csv = None
    probe_csv = None

    best_probe_mae = float("inf")
    recent: Dict[str, List[float]] = {}
    batch_idx = 0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm.tqdm(total=args.max_steps, initial=global_step, desc="Train FAR (Focused Action Refinement)")
    while global_step < args.max_steps:
        for batch in train_loader:
            batch_gpu = move_batch_to_device(batch, device)
            bridge = extract_far_features(
                vla, batch_gpu, frozen_base_action_head, proprio_projector, cfg,
            )
            outputs = model(bridge)
            actions_gt = batch_gpu["actions"].to(torch.float32)

            loss, logs = far_loss(outputs, actions_gt, cfg)
            (loss / args.grad_accumulation_steps).backward()
            batch_idx += 1
            if batch_idx % args.grad_accumulation_steps != 0:
                continue

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)

            for key, value in logs.items():
                recent.setdefault(key, []).append(float(value))
                if len(recent[key]) > args.log_freq:
                    recent[key].pop(0)

            if global_step % args.log_freq == 0:
                avg = {key: sum(values) / len(values) for key, values in recent.items()}
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[1]["lr"]
                msg = [
                    f"step={global_step}",
                    f"loss={avg['loss']:.4f}",
                    f"improve={avg.get('final_vs_base_improve_pct', 0.0):+.3f}%",
                    f"grip_acc={avg.get('grip_acc', 0.0):.4f}",
                    f"tdn={avg.get('token_delta_norm', 0.0):.3f}",
                    # Post-gate (what fusion sees) + Pre-gate (attention strength)
                    f"H_V={avg.get('H_V_contrib', 0.0):.3f}",
                    f"H_V_raw={avg.get('H_V_raw_contrib', 0.0):.3f}",
                    f"H_AQ={avg.get('H_AQ_contrib', 0.0):.3f}",
                    f"H_A={avg.get('H_A_contrib', 0.0):.3f}",
                    f"v_max={avg.get('v_attn_max', 0.0):.4f}",
                    f"v_ent_abs={avg.get('v_attn_entropy_abs', 0.0):.3f}",
                    f"gate={avg.get('gate_mean', 0.0):.3f}",
                ]
                emit_progress_line(" | ".join(msg))

                if global_step % 500 == 0:
                    emit_progress_line(
                        f"[AQ-diag step={global_step}] "
                        f"H_AQ={avg.get('H_AQ_contrib', 0):.3f} (target<0.40)  "
                        f"H_V={avg.get('H_V_contrib', 0):.3f} (target>0.28)  "
                        f"gap={avg.get('H_AQ_contrib', 0) - avg.get('H_V_contrib', 0):.3f}"
                    )

                row = {"step": global_step, "lr_encoder": lr_enc, "lr_head": lr_head, **avg}
                if train_csv is None:
                    train_csv = csv.DictWriter(train_log_file, fieldnames=list(row.keys()))
                    train_csv.writeheader()
                train_csv.writerow(row)
                train_log_file.flush()

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                ckpt_path = output_dir / f"far_step{global_step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, scheduler, global_step, cfg)
                emit_progress_line(f"Saved: {ckpt_path}")

            if args.probe_freq > 0 and global_step % args.probe_freq == 0 and len(probe_cache) > 0:
                probe_metrics = run_probe(
                    model, probe_cache, device, cfg,
                    base_model=vla,
                    frozen_base_action_head=frozen_base_action_head,
                    proprio_projector=proprio_projector,
                )
                parts = [f"[PROBE step={global_step}]"] + [
                    f"{k}={v:.5f}" for k, v in sorted(probe_metrics.items())
                ]
                emit_progress_line(" | ".join(parts))

                probe_row = {"step": global_step, **probe_metrics}
                if probe_csv is None:
                    probe_csv = csv.DictWriter(probe_log_file, fieldnames=list(probe_row.keys()))
                    probe_csv.writeheader()
                probe_csv.writerow(probe_row)
                probe_log_file.flush()

                cur_mae = probe_metrics.get("probe/final_pose_mae", float("inf"))
                if cur_mae < best_probe_mae:
                    best_probe_mae = cur_mae
                    torch.save(
                        {
                            "step": global_step,
                            "model": model.state_dict(),
                            "best_probe_mae": best_probe_mae,
                            "cfg": asdict(cfg),
                        },
                        output_dir / "far_best_probe.pt",
                    )
                    emit_progress_line(
                        f"New best probe final_pose_mae={best_probe_mae:.5f} at step={global_step}"
                    )

            if global_step >= args.max_steps:
                break

    progress.close()
    final_path = output_dir / "far_final.pt"
    save_checkpoint(final_path, model, optimizer, scheduler, global_step, cfg)
    train_log_file.close()
    probe_log_file.close()
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
