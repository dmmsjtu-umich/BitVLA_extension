"""FAR rollout evaluation on LIBERO.

This is a thin wrapper that invokes the project-root `scripts/eval_dsar_rollout.py`
(which has been verified to produce the 88.6% rollout number reported in the paper).
Rather than duplicating the 2000+ line rollout code, we delegate to the project-root
script and pass the FAR checkpoint path through its `--dsar_checkpoint` argument.
The project-root script detects FAR models via the `model_family: "dsar_v7_focus"`
field in the config JSON and dispatches to the correct predict_action function.

For most users, running `scripts/eval_dsar_rollout.py` directly (as shown in
README §10) is the simplest path.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]              # repo root
# (no longer needed — repo root is the reference)
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_dsar_rollout.py"


def main():
    p = argparse.ArgumentParser(description="FAR rollout wrapper around scripts/eval_dsar_rollout.py")
    p.add_argument("--base_checkpoint", type=str, required=True)
    p.add_argument("--far_checkpoint", type=str, required=True,
                   help="Path to FAR trained checkpoint (e.g. far_step10000.pt)")
    p.add_argument("--far_config", type=str, required=True,
                   help="Path to FAR config json (saved by train.py)")
    p.add_argument("--task_suite_name", type=str, default="libero_10")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONPATH",
        f"{REPO_ROOT}:{REPO_ROOT}/repos/BitVLA/openvla-oft/bitvla:{REPO_ROOT}/repos/LIBERO")

    if not EVAL_SCRIPT.exists():
        print(f"ERROR: eval script not found at {EVAL_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--base_checkpoint", args.base_checkpoint,
        "--dsar_checkpoint", args.far_checkpoint,
        "--dsar_config", args.far_config,
        "--action_mode", "gate_off",
        "--task_suite_name", args.task_suite_name,
        "--num_trials_per_task", str(args.num_trials_per_task),
        "--seed", str(args.seed),
        "--output_dir", args.output_dir,
    ]
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
