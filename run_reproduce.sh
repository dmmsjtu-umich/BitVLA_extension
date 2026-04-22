#!/usr/bin/env bash
# FAR (Focused Action Refinement) — train + rollout end-to-end.
#
# Prerequisites:
#   - conda env `bitvla` with BitVLA + LIBERO + prismatic installed
#   - checkpoints/bitvla/ft-libero-long-bf16 in place (BitVLA official)
#   - data/modified_libero_rlds/libero_10_no_noops prepared
#
# Runtime:
#   Train  ~1.7h on RTX 5090
#   Rollout ~3-5h for 500 episodes
set -e

export MUJOCO_GL=egl
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# (no longer needed)
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/repos/BitVLA/openvla-oft/bitvla:$REPO_ROOT/repos/LIBERO"

BASE_CKPT="$REPO_ROOT/checkpoints/bitvla/ft-libero-long-bf16"
OUT="$REPO_ROOT/outputs/far_main"
ROLLOUT="$REPO_ROOT/outputs/rollout_far_main"

echo "=== [1/2] TRAIN ==="
conda run -n bitvla --no-capture-output python "$REPO_ROOT/scripts/train.py" \
    --base_checkpoint "$BASE_CKPT" \
    --output_dir "$OUT" \
    --max_steps 10000 \
    --batch_size 8 --grad_accumulation_steps 2 \
    --probe_freq 500 --save_freq 1000 \
    --seed 7

if [ ! -f "$OUT/far_step10000.pt" ]; then
    echo "TRAIN FAILED — no far_step10000.pt"
    exit 1
fi

echo "=== [2/2] ROLLOUT ==="
mkdir -p "$ROLLOUT"
conda run -n bitvla --no-capture-output python "$REPO_ROOT/scripts/rollout.py" \
    --base_checkpoint "$BASE_CKPT" \
    --far_checkpoint "$OUT/far_step10000.pt" \
    --far_config "$OUT/far_cfg.json" \
    --task_suite_name libero_10 \
    --num_trials_per_task 50 \
    --seed 7 \
    --output_dir "$ROLLOUT"

echo ""
echo "=== DONE ==="
python3 -c "
import json
d = json.load(open('$ROLLOUT/results.json'))
print('FAR rollout: {:.1f}% ({}/{})'.format(
    d['success_rate']*100, d['total_successes'], d['total_episodes']))
"
