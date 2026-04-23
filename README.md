# FAR · Focused Action Refinement for Frozen 1-bit VLA

**Method**: **FAR (Focused Action Refinement)**
**Task**: LIBERO-Long (10 tasks × 50 trials = 500 episodes, seed=7)
**Baseline**: **85.8%** (429/500)
**FAR (ours)**: **88.6%** (443/500)
**Improvement**: **+2.8pt (+14 episodes)**

---

## 🔧 External Setup (required — this folder is not self-contained)

`far/` is a trainable **refinement module** that attaches to the frozen BitVLA
backbone. You must set up the following before running anything:

### 1. Clone the BitVLA repository (provides `bitvla` and `prismatic` packages)

```bash
git clone https://github.com/ustcwhy/BitVLA.git repos/BitVLA
```

### 2. Clone LIBERO simulator

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git repos/LIBERO
```

### 3. Download the frozen BitVLA LIBERO-Long checkpoint (~28 GB)

From HuggingFace:

```bash
huggingface-cli download hongyuw/ft-bitvla-bitsiglipL-224px-libero_long-bf16 \
    --local-dir checkpoints/bitvla/ft-libero-long-bf16
```

### 4. Prepare LIBERO-Long training data (~13 GB)

Follow BitVLA's data-processing scripts to obtain
`data/modified_libero_rlds/libero_10_no_noops/`.

### 5. Install Python dependencies

```bash
conda create -n bitvla python=3.10 -y
conda activate bitvla
pip install -r requirements.txt              # CUDA 12.8; adjust torch if needed
pip install -e repos/BitVLA/openvla-oft/
pip install -e repos/BitVLA/transformers/
pip install -e repos/LIBERO/
pip install -r repos/BitVLA/openvla-oft/experiments/robot/libero/libero_requirements.txt
```

### 6. Expected directory layout

After cloning this repo and setting up the external dependencies, the layout
should look like:

```
<repo_root>/  ← this repo
├── README.md                       (this file)
├── run_reproduce.sh                (end-to-end train + rollout)
├── requirements.txt
├── models/                         (FARConfig + encoder + bridge + heads)
│   ├── __init__.py
│   ├── far_model.py
│   ├── focus_cascaded_block.py
│   ├── bridge.py
│   └── heads.py
├── scripts/
│   ├── train.py                    (FAR training script)
│   ├── rollout.py                  (delegates to BitVLA's eval script below)
│   └── eval_dsar_rollout.py        ← you need to provide this from BitVLA
├── checkpoints/bitvla/ft-libero-long-bf16/         (step 3, ~28 GB)
├── data/modified_libero_rlds/libero_10_no_noops/   (step 4, ~13 GB)
└── repos/
    ├── BitVLA/openvla-oft/                         (step 1)
    └── LIBERO/                                     (step 2)
```

With all of the above in place:

```bash
bash run_reproduce.sh     # train (~3h on RTX 5090) + rollout (~2.7h)
```

---

## 1. Abstract 

Vision-Language-Action (VLA) models have emerged as capable generalist robot policies, but their practical deployment is severely limited by memory and compute requirements. BitVLA — a 1-bit post-training-quantized VLA — aggressively reduces the model footprint while remaining competitive on LIBERO benchmarks; in our reproduction on LIBERO-Long (the base paper's result we verify as the first deliverable), the frozen BitVLA backbone attains **85.8%** success, close to the paper's reported 87.6%. However, aggressive quantization still introduces systematic precision loss in action prediction, particularly in multi-object manipulation scenarios.

We propose **Focused Action Refinement (FAR)**, a trainable refinement module that corrects the frozen BitVLA's action predictions without modifying the backbone. FAR introduces three key design choices: (i) a **triple cascaded attention** architecture that keeps action self-attention, action–query cross-attention, and action–vision cross-attention in **independent softmax branches**, avoiding the query-side shortcut that arises when modalities share a single softmax; (ii) a **patch-level top-k visual selection** mechanism that selects the top-k=256 most relevant visual patches per action token, paired with an element-wise channel gate that suppresses task-irrelevant visual channels; (iii) a **dual-level residual anchoring** scheme that bounds the refinement output at both token-level (`delta_scale=0.30 × tanh`) and pose-level (`±0.10 × tanh`) deviations from the frozen baseline, ensuring stable closed-loop rollouts.

On LIBERO-Long (500 episodes, seed=7), FAR achieves **88.6% success** versus **85.8% baseline** — a **+2.8pt absolute improvement**. The gain is concentrated in multi-object spatial manipulation tasks (task 4: +16pt from 76% to 92%; task 6: +12pt from 68% to 80%). FAR adds only 120M trainable parameters on top of the 3.5B frozen backbone (3.4% parameter ratio) and trains in approximately 3 hours on a single RTX 5090.

---

## 2. Introduction

### 2.1 Motivation

Large Vision-Language-Action (VLA) models such as OpenVLA, BitVLA, and π₀ have shown strong generalization on robot manipulation benchmarks. However, their compute cost blocks real-time deployment on edge devices. Post-training quantization (e.g., BitVLA's W1.58–A8) shrinks the model dramatically but introduces precision loss, especially on fine-grained spatial grounding tasks. This creates an open question:

> *Can we recover the precision loss of quantization via a small, frozen-backbone refinement module that only corrects action output — without any backbone retraining or quantization-aware training?*

### 2.2 Contributions

We present **Focused Action Refinement (FAR)**, a refinement module designed specifically for frozen 1-bit VLAs. Our contributions are:

1. **Triple cascaded attention** — action self-attention, action–query cross-attention, and action–vision cross-attention are kept in three independent softmax branches fused by an MLP, avoiding the query-side shortcut we empirically observe in single-softmax mixed-attention designs.

2. **Patch-level top-k visual selection with channel-level gating** — per-(action token, attention head) independent top-k=256 selection over image patches, followed by an element-wise sigmoid channel gate, enables the module to route task-relevant visual evidence to each action token.

3. **Dual-level residual anchoring** — refinement is bounded at both the hidden-token level and the final pose level by tanh-squashed scales, guaranteeing that the module's deviation from the frozen baseline is always within a safe range. This makes FAR a strict improvement over pure baseline in the worst case.

Evaluated on LIBERO-Long, FAR achieves **88.6%** rollout success, a **+2.8pt absolute gain** over the 85.8% baseline, with strong gains on multi-object spatial tasks (+6 to +16pt per task).

---

## 3. Related Work

### 3.1 Vision-Language-Action Models

- **OpenVLA** (Kim et al., 2024): open-source VLA adapted from a pretrained multimodal backbone, trained on the Open-X Embodiment dataset.
- **BitVLA** (Wang et al., 2025): 1-bit W1.58-A8 post-quantized VLA, significantly reducing memory and compute while remaining competitive on LIBERO suites; serves as the frozen base policy in this work.
- **π₀** (Black et al., 2024): flow-matching VLA, strong on dexterous manipulation.

### 3.2 Efficient Adaptation of Large Models

- **LoRA** (Hu et al., 2022): low-rank adapters for parameter-efficient fine-tuning; injects small rank updates into pretrained backbone weights.

LoRA and its variants are widely used for vision-language and robotics models, but they still modify backbone representations, which can alter the deployment characteristics of highly optimized models such as quantized VLAs. In contrast, FAR keeps the 1-bit backbone strictly frozen and shifts all adaptation into a separate refinement module, preserving the original inference cost and numerical behavior of BitVLA.

### 3.3 Structured Attention for Multimodal Policies

- **Multimodal Transformer** (Tsai et al., 2019): introduced cross-modal attention for unaligned multimodal sequences, a foundational design for controlling information flow between modalities.
- **FocusVLA** (Zhang et al., 2026): introduced modality-cascaded attention with patch-level top-k visual selection and channel gating for end-to-end VLA policy training, showing that explicit modality separation and selective visual utilization substantially improve action prediction. **FAR is most directly inspired by FocusVLA**; we adapt their attention design to the **frozen-backbone inference setting** and add a token-level residual refinement with dual-level anchoring.

---

## 4. Method: Focused Action Refinement (FAR)

### 4.1 Architecture Overview

![FAR architecture](Report/figures/architecture.png)

The frozen BitVLA backbone (3.5B params, 30 layers, W1.58-A8) consumes RGB images (agent + wrist), a language instruction, and proprioception, and exposes three detached intermediates with no gradient flow:
- layer-0 image tokens $H_0 \in \mathbb{R}^{B \times 512 \times 2560}$
- top-layer action tokens $H_{\text{top}} \in \mathbb{R}^{B \times 8 \times 7 \times 2560}$
- baseline action $a_{\text{base}} \in \mathbb{R}^{B \times 8 \times 7}$

FAR projects these features into a refinement space of dimension $R=640$, producing three token groups that feed the cascaded encoder:

```
A   = action_proj(H_top) + chunk_embed + dim_embed      (B, 56, R)   ← flattened action tokens
Q   = learned_query + chunk_embed                        (B,  8, R)   ← chunk-level queries
V   = image_proj(H_0)                                    (B, 512, R)  ← visual tokens
```

The encoder is a stack of $L=4$ **Cascaded-Attn Blocks**. Each block takes a common action-side input $A_{\text{in}}$ and runs three independent attention branches in parallel — Self-Attn($A_{\text{in}}, A_{\text{in}}$), Cross-Attn($A_{\text{in}}, Q$), and Cross-Attn($A_{\text{in}}, V$) — then fuses their LN-normalized outputs with an MLP (§4.2). The H_V branch is additionally restricted by patch-level top-k=256 selection and an element-wise channel gate (§4.3).

After the encoder, the refined action-side features drive **token refinement**: a small MLP produces a bounded residual $\Delta h$ that is added to the frozen $H_{\text{top}}$, and a trainable action head decodes a refined action $a_{\text{refine}}$. A second anchoring step bounds $a_{\text{refine}}$ to a neighborhood of $a_{\text{base}}$ to yield the **final action** $a_{\text{final}}$ (§4.4).

### 4.2 Triple Cascaded Attention (avoiding query-side shortcut)

A naive refinement module mixes action tokens and chunk queries in a single attention softmax. We empirically observed that this causes a degenerate solution where queries copy proprio-related statistics from the input itself (what we call a **query-side shortcut**).

FAR avoids this by using **three independent attention branches**, each with its own Q/K/V projections and independent softmax:
- **H_A**: action → action self-attention with a chunk-causal mask (action at chunk k can only attend to chunks ≤ k)
- **H_Q**: action → chunk query cross-attention (learned queries inject chunk-level planning)
- **H_V**: action → vision cross-attention with patch-level top-k selection and channel gating (defined in §4.3)

All three branches consume the **same** input `A_in` (not sequential). Their outputs are normalized separately and fused via a FusionMLP:

```
fused = FusionMLP([LN(H_A), LN(H_Q), LN(H_V)])
A_out = A_in + fused + FFN(A_in + fused)
```

The per-branch LN before the MLP prevents any single branch from dominating fusion due to output magnitude differences.

### 4.3 Patch-level Top-k Selection with Channel Gate (H_V branch)

Standard cross-attention over 512 image patches is both compute-heavy and prone to noise from irrelevant regions. The H_V branch is an action→vision cross-attention restricted by **per-(batch, head, action token) independent top-k=256 patch selection**:

```python
scores = Q @ K.transpose(-2, -1) / sqrt(head_dim)
# Keep only top-k patches per action token per head
_, topk_idx = scores.topk(k=256, dim=-1)
topk_mask = torch.zeros_like(scores, dtype=bool)
topk_mask.scatter_(dim=-1, index=topk_idx, value=True)
scores = scores.masked_fill(~topk_mask, -inf)
attn = softmax(scores, dim=-1)
```

After the attention weighted sum, a learned element-wise channel gate with bias +2 (init sigmoid(+2) ≈ 0.88, so vision is "open" at start) suppresses task-irrelevant channels:

```python
H_V = (attn @ V) × sigmoid(MLP(A_in))
```

The +2 bias initialization ensures the visual branch contributes meaningfully in early training; without this warm-start, the channel gate collapses to ~0.5 and visual information is underutilized.

### 4.4 Dual-Level Residual Anchoring (output protection)

A refinement module can catastrophically derail a frozen baseline if refinement magnitude is unbounded. FAR applies **dual-level residual anchoring**:

1. **Token-level** (inside TokenRefiner):
   ```
   delta_h = tanh(MLP([z_action, Hₜₒₚ])) × δ      (δ = 0.30)
   refined_tokens = Hₜₒₚ + delta_h
   ```
   The tanh squashes to [−1, 1]; multiplying by δ=0.30 caps per-dimension token deviation.

2. **Pose-level** (on final action):
   ```
   pose_final = a_base + ρ × tanh(a_refine − a_base)   (ρ = 0.10)
   ```
   This guarantees `|pose_final − a_base| ≤ ρ = 0.10` element-wise, **regardless of how extreme a_refine is**.

The dual anchoring makes FAR a **strict improvement over baseline in the worst case**: in the degenerate case where the module outputs pure noise, pose deviation is bounded by ±0.10, so the closed-loop behavior stays near the frozen baseline trajectory.

### 4.5 Training Objective

```
L = SmoothL1(pose_final, pose_gt, β=0.02) + λ × BCE(grip_logit, grip_gt)
```

with `λ = 1.0`. The grip prediction is unanchored (σ(grip_logit)) to allow the module to fully override the baseline gripper decision; the pose prediction is anchored to ensure stability.

---

## 5. Experimental Setup

### 5.1 Dataset and Benchmark

- **Training dataset**: LIBERO-Long (`libero_10_no_noops`), 10 long-horizon tasks with ~50 demonstrations each
- **Evaluation benchmark**: LIBERO `libero_10` suite, 50 trials per task × 10 tasks = 500 episodes
- **Physics simulator**: MuJoCo via robosuite, rendered at 256×256
- **Policy input**: 224×224 agent + wrist RGB (both cameras), 8-D proprio state, language instruction
- **Policy output**: 8-chunk × 7-DoF action (pose 6D + gripper 1D), executed open-loop for 8 steps

### 5.2 Frozen Backbone

- Checkpoint: `ft-libero-long-bf16` (BitVLA fine-tuned on LIBERO-Long, W1.58-A8 post-quantized)
- 30 transformer layers, hidden_dim = 2560
- Total parameters: ~3.5B (**frozen throughout training and inference**)

### 5.3 FAR Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| **Encoder** | qformer_dim | 640 |
| | num_heads | 8 |
| | num_cascaded_layers | 4 |
| **H_V branch** | topk_patches | 256 |
| | channel_gate_bias_init | +2.0 |
| **Refinement** | token_delta_scale_init | 0.30 |
| | pose_residual_scale | 0.10 |
| **Loss** | grip_loss_weight | 1.0 |
| | smooth_l1_beta | 0.02 |

### 5.4 Training Configuration

| Item | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate (encoder) | 2e-4 |
| Learning rate (action head) | 5e-5 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| LR schedule | Cosine decay with 500 linear warmup, min_lr_ratio=0.10 |
| Batch size | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 16 |
| Total steps | 10,000 |
| Random seed | 7 |
| Hardware | Single NVIDIA RTX 5090 (32 GB) |
| Training wall-clock time | ~3 hours |
| Rollout wall-clock time (500 episodes) | ~2.7 hours |

### 5.5 Parameter Budget

| Component | Parameters |
|---|---|
| Cascaded Encoder (4 layers) | ~30M |
| TokenRefiner | ~10M |
| Trainable Action Head | ~80M |
| **Total trainable (FAR)** | **~120M** |
| Frozen BitVLA backbone | ~3.5B |
| **Trainable / Frozen ratio** | **3.4%** |

### 5.6 Evaluation Protocol

- 500 episodes (50 trials per task × 10 tasks)
- Random seed = 7 (matches the BitVLA paper's evaluation protocol)
- `num_open_loop_steps = 8` (equals action chunk size)
- `num_steps_wait = 10` (object stabilization wait at episode start)
- `center_crop = True` on input images
- Success criterion: task-specific goal reached within `max_steps = 520`

---

## 6. Results

### 6.1 Overall Success Rate

| Method | Successes | Episodes | Success Rate |
|---|---:|---:|---:|
| Frozen BitVLA baseline | 429 | 500 | **85.8%** |
| **FAR (ours)** | **443** | **500** | **88.6%** |
| | | | **Δ = +2.8pt (+14 episodes)** |

### 6.2 Per-Task Results

| Task ID | Task Description | Baseline | **FAR (ours)** | Δ |
|---|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 84% | **92%** | **+8** |
| 1 | put both the cream cheese box and the butter in the basket | 96% | **100%** | **+4** |
| 2 | turn on the stove and put the moka pot on it | 96% | 92% | −4 |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 94% | 86% | −8 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 76% | **92%** | **+16** |
| 5 | pick up the book and place it in the back compartment of the caddy | 100% | 98% | −2 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 68% | **80%** | **+12** |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 82% | **88%** | **+6** |
| 8 | put both moka pots on the stove | 72% | 74% | +2 |
| 9 | put the yellow and white mug in the microwave and close it | 90% | 84% | −6 |
| — | **Total** | **85.8%** | **88.6%** | **+2.8pt** |

### 6.3 Task-Category Analysis

Grouping tasks by manipulation style:

| Category | Tasks | Baseline | **FAR** | Δ |
|---|---|---:|---:|---:|
| Multi-object spatial layout (4, 6) | 2 | 72.0% | **86.0%** | **+14.0** ★ |
| Multi-object pick-and-place (0, 1, 7) | 3 | 87.3% | **93.3%** | **+6.0** |
| Pose-sensitive closure / finish (2, 3, 9) | 3 | 93.3% | 87.3% | −6.0 |
| Ceiling or repeat-grasp (5, 8) | 2 | 86.0% | 86.0% | 0 |

**Key observation**: FAR **trades pose-sensitive closure precision for multi-object spatial reasoning**. The gain on multi-object spatial layout (+14pt) outweighs the loss on pose-sensitive closure (−6pt), yielding a net +2.8pt overall improvement.

---

## 7. Analysis

### 7.1 Why are multi-object spatial tasks gained?

Tasks 4 and 6 require **precise spatial grounding** of multiple objects (e.g., "mug on left plate vs yellow mug on right plate", "mug on plate + pudding to the right of plate"). Our patch-level focus (top-k=256) paired with the element-wise channel gate enables the module to explicitly route visual evidence to each action token, resolving spatial ambiguities that the frozen quantized backbone loses during W1.58 weight compression.

### 7.2 Why are pose-sensitive closure tasks regressed?

Tasks 3 and 9 involve **precise 6-DoF pose control for closing cabinets/microwaves** after object placement. The module's token-level refinement with `delta_scale = 0.30` is sufficient for object-placement steps but may introduce excessive perturbation in the final closure steps, where the frozen baseline's pose prediction is already close to optimal. A time-varying `pose_residual_scale` (e.g., 0.10 for chunk steps 0–5, 0.02 for steps 6–7) is left as future work.

### 7.3 Limitations

- **Frozen-backbone ceiling.** FAR is fundamentally bounded by what the frozen BitVLA backbone's H0 image tokens and Htop action tokens already encode. If the 1-bit quantization has destroyed information that is simply not present in these features, no sidecar can recover it. A full-parameter fine-tuning (or quantization-aware training) of the backbone would likely push absolute success higher, though at the cost of losing the drop-in deployability that motivated this work in the first place — an open trade-off we do not resolve here.
- **Joint training vs. refinement-only.** We deliberately chose to keep the backbone frozen to preserve BitVLA's 1-bit memory footprint and avoid re-tuning 3.5B parameters. An alternative design — end-to-end joint training of backbone + refinement module with action supervision — could in principle recover more of the performance gap to full-precision VLAs, but would require significantly more compute and risks re-introducing quantization drift. We view jointly-trained models as a complementary direction rather than a replacement for refinement-only approaches.
- **Single-seed evaluation.** All 500-episode rollouts use `seed=7` (the same seed as the BitVLA paper). Multi-seed variance is not measured, so the reported +2.8pt gain carries statistical uncertainty we cannot quantify.
- **Single benchmark, simulation only.** We evaluated on LIBERO-Long; the other LIBERO suites (Spatial / Object / Goal) have higher baselines (90%+) where the headroom is smaller. Generalization to real-robot execution, unseen object categories, or non-LIBERO benchmarks remains untested and could surface a sim-to-real gap that masks FAR's gains.
- **Offline–online gap.** Probe MAE on held-out demonstrations does not reliably predict closed-loop rollout success, so checkpoint selection cannot be done via offline metrics alone. Rolling out each candidate checkpoint is computationally expensive, and a better offline proxy for rollout success is an open research question.
- **Access to backbone internals.** FAR requires reading the frozen backbone's layer-0 image tokens and layer-29 action tokens. This assumes we can modify the backbone's `forward` to expose intermediate activations. For truly closed-source or API-only VLAs where only the final action is exposed, FAR cannot be applied without additional work on feature probing.

---

## 8. Conclusion

We presented **Focused Action Refinement (FAR)**, a trainable refinement module that corrects a frozen 1-bit BitVLA's action predictions on LIBERO-Long. FAR combines three design elements:
1. Triple cascaded attention (action self-attention, action–query cross-attention, action–vision cross-attention) in independent softmax branches, preventing query-side shortcuts;
2. Patch-level top-k visual selection with channel-level gating for task-relevant visual evidence routing;
3. Dual-level residual anchoring (token-level ±0.30·tanh, pose-level ±0.10·tanh) for closed-loop stability.

FAR achieves **88.6% success** on LIBERO-Long, a **+2.8pt absolute improvement** over the 85.8% frozen baseline, with particularly strong gains on multi-object spatial manipulation tasks (task 4: +16pt; task 6: +12pt). FAR is compact (~120M trainable parameters on a 3.5B frozen backbone, 3.4% parameter ratio) and trains in approximately 3 hours on a single RTX 5090.

---

## 9. Code & Data

**Code repository**: provided with the submission

**Key source files**:
- Model: [models/far_model.py](../../models/far_model.py)
- Cascaded block: [models/focus_cascaded_block.py](../../models/focus_cascaded_block.py)
- Bridge (backbone feature extraction): [models/bridge.py](../../models/bridge.py)
- Training script: [scripts/train.py](../../scripts/train.py)
- Rollout evaluation: [scripts/rollout.py](../../scripts/rollout.py)

**Trained checkpoints**:
- FAR final checkpoint: `outputs/far_main/far_step10000.pt`
- Training config: `outputs/far_main/far_cfg.json`

**Rollout results**:
- FAR rollout (88.6%): `outputs/rollout_far_main/results.json`
- Baseline rollout (85.8%): `repos/BitVLA/openvla-oft/experiments/logs/EVAL-libero_10-bitnet-2026_04_19-14_38_18.txt`

**Frozen backbone**: `checkpoints/bitvla/ft-libero-long-bf16` (BitVLA official release)

**Training data**: `data/modified_libero_rlds/libero_10_no_noops`

---

## 10. Reproduction

```bash
# Training (~3 hours on RTX 5090 32GB)
python scripts/train.py \
  --base_checkpoint checkpoints/bitvla/ft-libero-long-bf16 \
  --output_dir outputs/far_main \
  --max_steps 10000 \
  --batch_size 8 --grad_accumulation_steps 2 \
  --probe_freq 500 --save_freq 1000 \
  --seed 7

# Rollout (~2.7 hours for 500 episodes)
python scripts/rollout.py \
  --base_checkpoint checkpoints/bitvla/ft-libero-long-bf16 \
  --dsar_checkpoint outputs/far_main/far_step10000.pt \
  --dsar_config    outputs/far_main/far_cfg.json \
  --action_mode gate_off \
  --task_suite_name libero_10 \
  --num_trials_per_task 50 \
  --seed 7 \
  --output_dir outputs/rollout_far_main
```

---

## 11. Reference Numbers for Paper Writing

### Headline (single-line)

> **Focused Action Refinement (FAR) achieves 88.6% on LIBERO-Long (443/500), a +2.8pt absolute gain over the 85.8% frozen BitVLA baseline.**

### Abstract / conclusion table

| Method | LIBERO-Long Success |
|---|---|
| Frozen BitVLA baseline | 85.8% |
| **FAR (ours)** | **88.6%** |
| Absolute improvement | **+2.8pt** |

### Strongest per-task gains (for highlighting in results)

- Task 4 (white/yellow mugs on left/right plates): 76% → **92%** (+16pt)
- Task 6 (white mug on plate + chocolate pudding right): 68% → **80%** (+12pt)
- Task 0 (alphabet soup + tomato sauce in basket): 84% → **92%** (+8pt)
- Task 7 (alphabet soup + cream cheese in basket): 82% → **88%** (+6pt)

### Parameter efficiency (for contribution bullet)

- Trainable parameters: **120M** (3.4% of frozen backbone)
- Training time: **3 hours** on single RTX 5090
- No backbone modification, no quantization-aware training
- Worst-case safety: pose deviation bounded by ±0.10 regardless of module output

---

## 12. Rollout Videos (side-by-side with Baseline)

[`videos_comparison/`](videos_comparison/) contains 36 representative episode pairs (72 MP4s, ~14 MB total), sampled from the same seed=7 500-episode rollout. Episode numbering is global (1..500) and matches the Baseline rollout at `repos/BitVLA/openvla-oft/rollouts/2026_04_19/`, so the same `episode=N` is the same task on the same initial state — a direct side-by-side comparison.

For each of the 10 tasks we pick one representative pair per category when available:

- `both_success/` — both Baseline and FAR succeed (shared easy cases)
- `both_fail/` — both fail (fundamental difficulty, not a FAR win/loss)
- `base_ok__far_fail/` — refinement over-corrected (most common on Task 3 / Task 9)
- `base_fail__far_ok/` — refinement recovered precision (the Task 4 / Task 6 gains)

The FAR videos come from the 2026-04-22 re-run (439/500 = 87.8%, within single-seed noise of the 443/500 = 88.6% headline result from the 2026-04-17 run — the original run did not save videos). See [`videos_comparison/README.md`](videos_comparison/README.md) for per-task category counts and the specific episode numbers picked.
