# FAR vs Baseline: per-task representative rollouts

500-episode rollout on LIBERO-Long, seed=7.

- Baseline: frozen BitVLA `ft-libero-long-bf16` (429/500 = 85.8%), run 2026-04-19.
- FAR: `outputs/v7_focus_full_10k/v7_focus_best_probe.pt` on the same frozen backbone (439/500 = 87.8% on this run, 443/500 = 88.6% on the original 2026-04-17 run; both within single-seed noise).

Episodes are globally numbered 1..500 using `episode = task_id*50 + episode_idx + 1`, so `episode=N` in `2026_04_19/` (baseline) and `2026_04_22/` (FAR) refer to the **same task on the same initial state** — a direct side-by-side comparison.

Each folder below holds one representative pair per category when available.

## Categories

- `both_success`: both Baseline and FAR succeed — shows they solve the same scene.
- `both_fail`: both fail — fundamental difficulty, not a FAR win/loss.
- `base_ok__far_fail`: Baseline succeeds, FAR fails — cases where refinement over-corrects (matches the per-task regressions on Task 3 / Task 9).
- `base_fail__far_ok`: Baseline fails, FAR succeeds — cases where refinement recovers precision (matches the gains on Task 4 / Task 6).

## Per-task summary

| Task | Description | both_ok | both_fail | base_ok_far_fail | base_fail_far_ok |
|---|---|---:|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in | 42 | 4 | 0 | 4 |
| 1 | put both the cream cheese box and the butter in th | 46 | 1 | 2 | 1 |
| 2 | turn on the stove and put the moka pot on it | 44 | 1 | 4 | 1 |
| 3 | put the black bowl in the bottom drawer of the cab | 44 | 1 | 3 | 2 |
| 4 | put the white mug on the left plate and put the ye | 35 | 3 | 3 | 9 |
| 5 | pick up the book and place it in the back compartm | 48 | 0 | 2 | 0 |
| 6 | put the white mug on the plate and put the chocola | 27 | 4 | 7 | 12 |
| 7 | put both the alphabet soup and the cream cheese bo | 37 | 3 | 4 | 6 |
| 8 | put both moka pots on the stove | 31 | 9 | 5 | 5 |
| 9 | put the yellow and white mug in the microwave and  | 40 | 0 | 5 | 5 |
