# Glucose Transformer Hyperparameter Optimization Plan

## Scope

Build and run the experiment entirely from the FabricPC fork. The
glucose-forecasting repository is a one-time migration/parity reference, not a
runtime dependency. Existing 64-step runs are exploratory baselines and are not
architecture-equivalent GluMind-Uni comparisons.

The public example is named `glucose_transformer`:

- Python entry point: `examples/glucose_transformer.py`
- Console command: `glucose-transformer`
- Configs, run directories, and reports use the same name.

## 0. Reference-backend audit (glucose-forecasting/backends/pc)

The existing backend (train.py, data.py, config.py, cli.py) plus the
sugar_uni_pc.py model builder serve as a migration reference. The following
issues must be fixed in the FabricPC example — not ported as-is.

### Training loop problems (fixed in FabricPC example)

- **No learning rate schedule:** uses a flat `optax.adam(lr)` with no warmup,
  cosine decay, or weight decay.
  → *Fixed:* warmup + cosine decay schedule (`optax.warmup_cosine_decay_schedule`).
- **No gradient or norm clipping** on the optimizer side (PC norm clip on
  latent inference exists via `InferenceSGDNormClip`, but weight gradients
  are unclipped).
- **No early stopping:** runs all configured epochs regardless of val
  performance or instability.
  → *Fixed:* patience-based early stopping (default 4 epochs without improvement).
- **No instability detection:** no NaN/Inf energy checks, no energy-explosion
  guard during training. A diverging trial wastes the full epoch budget.
  → *Fixed:* immediate stop on NaN/Inf loss per batch.
- **Massive code duplication:** `train_pc` and `train_backprop` are 90%
  identical (>100 lines each); only the step function and loss variable name
  differ.
  → *Fixed:* single `train_single()` accepts either mode via a callback pattern.
- **No checkpoint resumability:** `_save_epoch_checkpoint` writes atomic
  checkpoints but no resume-from-checkpoint path exists.
  → *Fixed:* full training state checkpoint saved after each epoch; `--resume`
  flag restores params, optimizer state, RNG, epoch counter, and best MAE.
- **Per-batch metrics missing:** only prints training loss, not glucose-scale
  MAE/MARD during training.
  → *Fixed:* per-batch forward pass computes and logs MAE (mg/dL) and MARD (%).

### Configuration problems

- **Dual config system:** `TrainConfig` (dataclass in train.py) and
  `PcTrainSettings` (Pydantic in config.py) overlap but aren't unified.
  `PcTrainSettings` validates but is not used by the training loop.
- **No YAML-driven trial configuration:** tuning would need to wire Optuna
  around the dataclass manually; the Pydantic model supports YAML but
  doesn't connect to training.
- **Hardcoded optimizer:** always Adam, no AdamW or configurable weight decay.

### Model / node problems

- **Nodes live in glucose-forecasting:** `ContinuousEmbeddingNode`,
  `MultiScaleMhaResidualNode`, and `RegressionOutputNode` are defined in
  `models/sugar_uni_pc.py` inside glucose-forecasting. They must be moved
  into `fabricpc/nodes/` (or at minimum into the FabricPC example).
- **Single `forward` code path for `MultiScaleMhaResidualNode`:** attention
  QKV projection, RoPE, multi-scale pooling, and residual fusion are all in
  one `forward` method (~50 lines). This makes it hard to test or reuse
  individual components.
- **Output node uses seq_len × embed_dim flattening:** `RegressionOutputNode`
  flattens the full sequence, creating a parameter count proportional to
  `seq_len * embed_dim * embed_dim`. At seq_len=128, embed_dim=32 (the
  target geometry) that's a 131 K dense layer. This is the
  same design as PyTorch GluMind-Uni and should be kept for parity, but
  noted as an efficiency concern for future work.

### Architecture differences from PyTorch GluMind-Uni

| Aspect | PyTorch GluMind-Uni | PC sugar_uni_pc |
|---|---|---|
| Positional encoding | Sinusoidal (additive) | RoPE (rotary, inside Q/K) |
| LayerNorm placement | Post-norm (LN after residual add) | Pre-norm (LN before attention) |
| Dropout | 0.1 throughout | None (no dropout in JAX PC path) |
| Attention residual | `LN(x + attn(x))` then pool from that | Pool from `LN(x)`, add `skip + high + up2 + up4` |
| FFN structure | Single `nn.Sequential(Linear, GELU, Dropout, Linear)` + residual + LN | Decomposed into `LnMlp1Node` (LN + Linear + GELU) → `Mlp2ResidualNode` (Linear + residual) as separate PC nodes |
| Scale fusion | `high + dropout(up2) + dropout(up4)`, then FFN with its own residual | `skip + high + up2 + up4` (residual via skip slot) |
| Multi-scale attention input | LN + high-res attn + residual first, then pool that for low-res | LN once, then all three scales from the same pre-norm input |

These differences are deliberate FabricPC/PC decomposition choices.  Document
them but do not attempt to eliminate them — the controlled comparison
evaluates PC-vs-backprop on the *same* FabricPC graph, not FabricPC-vs-PyTorch.

### Data pipeline (reusable)

The data module (`data.py`) is clean and reusable:
- `load_glucose_csv` — column-name normalization
- `split_by_sequence` — leak-free sequence-ID split with optional
  `Recommended Split` column
- `build_sliding_windows` — stride-based `(X, Y)` array construction
- `normalize_glucose` — min-max on train statistics only
- `GlucoseWindowLoader` — batch iterator yielding `{"x": ..., "y": ...}`

Port these utilities into the FabricPC example as a self-contained data module.

## 1. Establish a correct shared graph

- Port the intended GluMind-Uni geometry into FabricPC: `seq_len=128`,
  `depth=3`, `embed_dim=32`, `num_heads=4`, `mlp_dim=128`, `horizon=12`.
  The canonical SugarOne/GluMind geometry is 128 input steps at 5-min
  cadence (~10.67 hours context), horizon=12 (60 min). The PC backend's
  `seq_len=64` default was an unintentional deviation — the NeuralForecast
  backend, evaluation adapter, and holdout protocol all enforce 128/12.
- Move `ContinuousEmbeddingNode`, `MultiScaleMhaResidualNode`, and
  `RegressionOutputNode` from glucose-forecasting into FabricPC — either
  under `fabricpc/nodes/` as reusable regression/continuous-input nodes or
  directly in the example module if they are too glucose-specific.
- Keep the graph builder (`create_sugar_uni_pc` → renamed
  `create_glucose_transformer`) under `examples/` or `fabricpc/models/`.
- Give the flatten projection and final projection separate,
  fan-in-appropriate initializers. The reference uses
  `NormalInitializer(std=sqrt(1/embed_dim))` for the output — keep this.
- Document deliberate differences from PyTorch GluMind-Uni (see table above).
- Add forward/shape and one-real-batch smoke checks. Do not launch long trials
  until PC and backprop execute the same graph successfully.

## 2. Define fair PC-vs-backprop comparisons

- **Controlled comparison:** identical graph, initialization seed, shuffled
  batch order, optimizer family, schedule, update budget, validation cadence,
  and data windows. Only PC latent-inference parameters differ because
  backprop has no equivalent.
- **Best-achievable comparison:** tune PC and FabricPC backprop separately with
  equal trial counts and equal total optimizer-update budgets.
- Treat PyTorch GluMind-Uni as an external implementation reference, not the
  causal PC-vs-backprop comparison; report framework and block differences.

## 3. Self-contained data (implemented)

- The prepared dataset `livia_sugar_one_ready.csv` (139K rows, Apache-2.0)
  is bundled at `examples/data/livia_sugar_one_ready.csv`.
- `examples/glucose_data.py` loads the bundled CSV directly; falls back to
  downloading `livia_mini.csv` from HuggingFace if the bundled file is absent.
- Leak-free splitting uses the `Recommended Split` column when present,
  otherwise random split by `sequence_id`.
- Sliding windows: `(N, seq_len=128, 1)` input → `(N, horizon=12)` target.
- Min-max normalization on train statistics only.
- No glucose-forecasting imports required.

## 4. Training protocols

- Train for up to 30 epochs with early stopping (patience=4 epochs without
  val MAE improvement). Stop immediately on NaN/Inf (instability).
- Warmup + cosine decay LR schedule sized to the total epoch budget:
  `total_steps = epochs × batches_per_epoch`.
- Save a full checkpoint (params, optimizer state, RNG, epoch, best MAE)
  after each epoch for crash recovery. Use `--resume` to continue from the
  last stable checkpoint.
- Per-batch logging: training loss, MAE (mg/dL), MARD (%) on the current
  training batch via a forward pass, enabling real-time monitoring.
- End-of-epoch validation: MAE, RMSE, MARD on the full validation set.
- History written incrementally to `history.csv` after each epoch.
- Primary objective: validation MAE in mg/dL. Secondary metrics: RMSE and MARD.
- Keep test splits sealed until configurations are selected; evaluate on
  held-out test set only using the best checkpoint after training completes.

## 5. Stage the search

1. **Architecture screen in both modes:** depth `{2,3,4}`, width `{32,64}`,
   heads `{4,8}` when divisible, FFN ratio `{2,4}`, with sequence length fixed
   at 128 (matching the canonical SugarOne/GluMind 128/12 geometry). Use
   paired seeds and prune unstable trials.
2. **PC dynamics search:** starting point `lr=0.02`, `eta_infer=5e-5` (high
   weight LR, conservative latent inference). Search `lr` log-range
   `5e-3..0.05`, `eta_infer` log-range `1e-5..5e-4`, `infer_steps` from
   `max(depth*3+2, 8)` through 24, norm clip `{0.5,1,5}`, weight-init
   standard deviation `0.01..0.03`, and a batch size selected to fit CUDA.
3. **Backprop optimizer search:** `lr` log-range `1e-5..3e-4`, weight decay
   `{0,1e-5,1e-4,1e-3}`, and gradient clip `{0.5,1,5}`, using the same
   architecture candidates and update budget.
4. **Confirmation:** run at least three paired seeds on Livia and Loop-dev
   validation sets. Select robust configurations by mean MAE while reporting
   variability and instability.
5. **Final evaluation:** evaluate selected PC and backprop checkpoints once on
   identical held-out windows. Report MAE, RMSE, MARD, runtime, peak GPU
   memory, parameter count, and update count.

## 6. Automated instability and restart loop

- Run each trial as a bounded child process with one active process group, a
  unique run directory, and a resumable Optuna journal.
- Monitor validation summaries, process exit, checkpoint timestamps, CUDA OOM
  messages, missing progress, and non-finite metrics.
- Apply deterministic early-stop rules:
  - stop immediately on NaN/Inf, CUDA OOM, missing progress, or energy greater
    than 10 times the recent finite median;
  - after at least three validation checks, stop when MAE regresses by more
    than 10% from the trial best for two consecutive checks;
  - stop for no meaningful improvement (`min_delta=0.25 mg/dL`) after four
    validation checks;
  - prune a short pilot that does not improve or remain stable across its first
    three checks.
- On instability, atomically preserve the last/best checkpoint, resolved YAML,
  metrics, failure reason, and stderr.
- Terminate the complete trial process group, verify that only that trial
  disappeared from `nvidia-smi`, and immediately launch the next Optuna
  suggestion.
- Do not reuse unstable model or optimizer state. Resume only interrupted,
  healthy trials from their atomic checkpoint.
- A CUDA OOM retry may halve batch size once. Repeated OOM or process-launch
  failure prunes the trial instead of looping indefinitely.
- Use validation events as the primary wake signal with a conservative
  fallback heartbeat. Never run duplicate watchers or duplicate trials.
- Stop after three consecutive infrastructure failures and report the blocker.
  Model-quality pruning does not count as infrastructure failure.
- Automatically enqueue full update-budget confirmation when a configuration
  passes the three-check pilot criterion.

## 7. Implementation map

### Adapt existing FabricPC infrastructure

- **`fabricpc/tuning/bayesian_tuner.py`** is autoregressive/perplexity-only.
  Either generalize it to accept a pluggable objective (MAE for regression) and
  a pluggable training loop, or write a parallel `RegressionTuner` that reuses
  the Optuna journal, two-phase search, and divergence-guard patterns. The
  energy-based divergence guard and Hyperband pruning structure are directly
  reusable.
- **`fabricpc/training/train.py`** (PC) and **`train_backprop.py`** provide
  `train_step` / `train_step_backprop` — these are the correct low-level
  step functions. The glucose example should call them directly rather than
  wrapping them in another duplicate loop. Build a single unified
  **step-based training loop** that accepts either step function via a
  callback, replacing the duplicated `train_pc` / `train_backprop` in the
  reference backend.
- **`fabricpc/models/transformer.py`** (`create_deep_transformer`) is the
  pattern to follow for the glucose model builder. The new
  `create_glucose_transformer` follows the same structure: node list, edge
  list, task map, inference config, muPC config.

### New files

| File | Purpose |
|---|---|
| `examples/glucose_transformer.py` | Entry point: data download/prep, model build, train, evaluate, compare. CLI via Typer or argparse. |
| `examples/glucose_transformer_tuning.py` | Two-phase Optuna search following `transformer_tuning.py` patterns. |
| `examples/glucose_data.py` | Self-contained data module: download, CSV prep, sliding windows, normalization, loader. No glucose-forecasting imports. |
| `examples/glucose_model.py` | `create_glucose_transformer` graph builder + node definitions (ContinuousEmbeddingNode, MultiScaleMhaResidualNode, RegressionOutputNode). |

### Training loop design

The unified training loop (`train_single` in `glucose_transformer.py`):

1. Accepts a mode (`pc` or `backprop`) and dispatches to the appropriate
   JIT-compiled step function — `train_step` (PC, returns energy) or
   `train_step_backprop` (backprop, returns MSE loss).
2. Trains for up to `--epochs` (default 30) full passes over the training set.
3. Logs per-batch: training loss, MAE (mg/dL), MARD (%) via a forward pass
   on the current training batch.
4. Validates at the end of each epoch: MAE, RMSE, MARD on the full val set.
5. Early stops after `--patience` epochs without val MAE improvement.
6. Stops immediately on NaN/Inf loss (instability detection).
7. Saves a full checkpoint after each epoch (params, optimizer state, RNG,
   epoch, best MAE) for crash recovery. `--resume` restores training state.
8. Writes `history.csv` incrementally (one row per epoch), `config.json`
   with full hyperparameters and results, and `best_params.pkl` for the
   best-val-MAE checkpoint.

### CLI commands

Expose concise `glucose-transformer` commands for:

| Subcommand | Action |
|---|---|
| `prepare` | Download and prepare Livia data, write cached arrays |
| `train` | Train a single configuration (PC or backprop) |
| `tune` | Run two-phase Optuna search |
| `evaluate` | Evaluate a checkpoint on held-out test windows |
| `compare` | Same-graph PC-vs-backprop comparison with paired seeds |

### Artifacts per trial

Store each trial's resolved YAML, seed, git revision, metric history, timing,
memory, checkpoint, and prune reason. Disable JAX CUDA preallocation, reuse
compiled functions, and avoid clearing JIT caches between validation checks.

## 8. Acceptance criteria

- PC and backprop consume byte-identical train/validation windows and paired
  initial parameters.
- A clean FabricPC checkout can download and prepare Livia data and run the
  complete comparison without a glucose-forecasting checkout.
- No result is compared by epoch count; every result includes optimizer update
  count.
- A selected configuration improves or remains stable across its first three
  validation checks and survives the full confirmation budget without
  catastrophic regression.
- Results are reproducible from checked-in YAML and a resumable Optuna journal.
- An injected unstable trial is pruned, unrelated GPU processes remain
  untouched, the reason is recorded, and exactly one replacement trial starts.
- The final report separates controlled same-graph results, separately
  optimized same-graph results, and the external PyTorch GluMind-Uni reference.
