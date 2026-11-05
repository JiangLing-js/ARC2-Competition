# Three-Wave Execution-Verified Program Synthesis with Lossless Hybrid Mamba-Attention Acceleration (`ARC2-Competition`)

**ARC Prize 2026 Official 240-Task Semi-Private Leaderboard:** **`36.25%`** (Submission `#57629846`, **Silver Medal**, Rank 135 / 3,627, Top 3.72%) & **`36.11%`** (Submission `#57903737`, `47.92%` on `Public-48`).

This repository contains the minimal core source code for our six-stage ARC-AGI-2 architecture designed for an air-gapped $4\times\text{NVIDIA L4}$ ($96\text{ GB}$ VRAM) 12-hour compute envelope.

---

## Repository Structure

### Stage 1: 2D Vision Transformer with Canvas Framing & 4-Connected BFS Object Positional Encoding (`35.38%` Pass@2 on $N=65$)
- [`src/`](src/): Core 2D Vision Transformer (`vit5_arc_encoder.py`, `vit5_dense_decoder.py`, `vit5_hybrid_model.py`, `vit5_layers.py`), all-color 4-connected BFS Object Positional Encoding (`vitarc_object_ids.py`, `vitarc_position_ids.py`, `vitarc_2d_rpe.py`), and Test-Time Training loops (`vit5_global_ttt_evaluation.py`, `vit5_per_task_ttt.py`).
- [`scripts/`](scripts/): Dataset preparation and Stage-1 training/TTT entrypoints (`prepare_arc_agi_datasets.py`, `run_vit5_clean_global_dense_colab.py`, `run_vit5_global_ttt_evaluation_colab.py`, `run_vit5_per_task_ttt_colab.py`).

### Stage 2: 16-Token Autoregressive Grid Transduction with Rank-256 `rsLoRA` TTT (`30.97%` Official LB `#56812471` & `30.28%` `#56537547`)
- [`experiments/nvarc_aimo_v1_reproduction_20261003/verified_output/`](experiments/nvarc_aimo_v1_reproduction_20261003/verified_output/): 16-token grid head, 128-step rank-256 `rsLoRA` per-task adaptation, and exact $D_8$ dihedral DFS candidate decoder (`arc_decoder.py`, `arc_loader.py`, `arc_solver.py`, `starter.py`).
- [`experiments/nvarc_equal_budget_20260921/runtime/`](experiments/nvarc_equal_budget_20260921/runtime/): Equal-budget $64+64$ split-view `rsLoRA` TTT implementation (`equal_budget_helpers.py`, `selection_helpers.py`, `arc_solver.py`).
- [`experiments/arc_task_conditioning_scaled_20260928/`](experiments/arc_task_conditioning_scaled_20260928/) & [`experiments/arc_meta_ttt_20260924/`](experiments/arc_meta_ttt_20260924/): Controlled ablations for FP32 task-conditioned Sigmoid gates (`task_gates.py`, `#56694740`) and offline SFT/DPO priors (`train_meta_initializer.py`, `preference_pipeline.py`, `#56588652`).

### Stages 3 & 4: Two-Stage Self-Correction Trajectory Distillation, Paired Bootstrap Audit & Per-Turn Horizon Regularization
- [`experiments/arc_gr48k_pi_stage2_sft_20261022/`](experiments/arc_gr48k_pi_stage2_sft_20261022/): Stage-2 multi-turn hypothesis-falsification trajectory curation and serving (`prepare_stage2.py`, `serve_stage2.py`).
- [`experiments/arc_stage_n_posttrain_20261029/`](experiments/arc_stage_n_posttrain_20261029/): Third/fourth-round SFT and shared-prefix code DPO builders plus 10,000-replicate paired bootstrap saturation audit (`build_stage_n_episodes.py`, `build_stage_n_sft_rows.py`, `build_stage_n3_dpo_pairs.py`, `build_stage_n4_rows.py`, `audit_r15_r17_and_lm.py`).

### Stages 5 & 6: `Speed4` Lossless Hybrid Mamba-Attention Inference Engine (`505–530 tok/s`) & Three-Wave Execution-Verified Synthesis (`36.25%` `#57629846` & `36.11%` `#57903737`)
- [`experiments/arc_tool_runtime_fix_20261023/vllm_patch/`](experiments/arc_tool_runtime_fix_20261023/vllm_patch/): Dual-bucket 24-slot Mamba recurrent-state pinning, `StarShm` lock-free POSIX `/dev/shm` TP=4 all-reduce, `TritonAttn v2`, `SmallTopK v4`, and `FA+16` EAGLE-1 speculative witness-state handoff (`install_mamba_pin_patch.py`).
- [`experiments/arc_tool_runtime_fix_20261023/deploy_candidate/`](experiments/arc_tool_runtime_fix_20261023/deploy_candidate/): $4\times\text{NVIDIA L4}$ server deployment and NCCL/SHM autotuning (`deployment.py`, `autotune_nccl.py`, `stage_model.py`).
- [`experiments/arc_tool_runtime_fix_20261023/harness/`](experiments/arc_tool_runtime_fix_20261023/harness/): Three-Wave execution-verified solver (`agent.mjs`, `panel.mjs`, `bridge.py`, `candidate-pool.mjs`, `kv-scheduler.mjs`, `time-governor.mjs`), implementing Wave-1 verified-consensus early exit, Wave-2 coordinate-difference witness repair with selective Group-B gating, per-turn $16\text{K}/22\text{K}$ thinking caps, and Wave-3 geometric transposition $T(X)=X^\top$.
