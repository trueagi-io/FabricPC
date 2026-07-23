"""Glucose Transformer — PC and backprop forecasting on the Livia dataset.

Usage:
    uv run glucose-transformer                          # PC mode, 30 epochs
    uv run glucose-transformer --mode backprop          # backprop mode
    uv run glucose-transformer --mode compare           # both modes, CUDA cleanup between
    uv run glucose-transformer --epochs 10 --patience 3
    uv run glucose-transformer --resume                 # resume from last checkpoint
    uv run glucose-transformer --platform cpu
"""
import os
import sys


def _init_jax_platform():
    """Ensure CUDA is used when available, unless user requests CPU."""
    platform = None
    for i, arg in enumerate(sys.argv):
        if arg == "--platform" and i + 1 < len(sys.argv):
            platform = sys.argv[i + 1]
    if platform is None:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                platform = "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    if platform:
        os.environ["JAX_PLATFORMS"] = platform


_init_jax_platform()

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import csv
import gc
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.graph_initialization import initialize_params
from fabricpc.training.train import train_step
from fabricpc.training.train_backprop import train_step_backprop, compute_forward_pass
from fabricpc.core.inference import InferenceSGDNormClip

from examples.glucose_data import prepare_data
from examples.glucose_model import create_glucose_transformer


def parse_args():
    p = argparse.ArgumentParser(description="Glucose Transformer (FabricPC)")
    p.add_argument("--mode", choices=["pc", "backprop", "compare"], default="pc")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed_dim", type=int, default=32)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--mlp_dim", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--max_updates",
        type=int,
        default=None,
        help="Optional optimizer-update budget for pilot runs",
    )
    p.add_argument("--lr", type=float, default=3.2753170973521557e-3)
    p.add_argument("--lr_backprop", type=float, default=1e-3,
                   help="Learning rate for backprop mode (default: 1e-3)")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument(
        "--decay_steps",
        type=int,
        default=None,
        help="Advanced LR decay override in optimizer updates",
    )
    p.add_argument(
        "--decay_epochs",
        type=int,
        default=None,
        help="LR decay horizon in epochs (default: full epoch budget)",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eta_infer", type=float, default=1.4435783212385837e-5)
    p.add_argument("--infer_steps", type=int, default=19)
    p.add_argument("--max_infer_norm", type=float, default=1.0)
    p.add_argument("--weight_init_std", type=float, default=0.02186191083483616)
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Global gradient norm clipping (default: 0.5)")
    p.add_argument(
        "--include_output_scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply muPC scaling to the regression output (default: enabled)",
    )
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="runs/glucose_transformer")
    p.add_argument("--log_every", type=int, default=1,
                   help="Log training loss every N batches within an epoch")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the last checkpoint")
    p.add_argument("--platform", type=str, default=None,
                   help="JAX platform: cuda, cpu (default: cuda if available)")
    return p.parse_args()


def evaluate(params, structure, loader, rng_key, g_min, g_max):
    """Feedforward eval returning glucose-scale MAE, RMSE, MARD."""
    glucose_range = max(g_max - g_min, 1e-8)
    total_se, total_ae, total_are, n = 0.0, 0.0, 0.0, 0

    @jax.jit
    def _eval_batch(p, batch, key):
        state = compute_forward_pass(p, structure, batch, key)
        preds = state.nodes[structure.task_map["y"]].z_mu * glucose_range + g_min
        targets = batch["y"] * glucose_range + g_min
        ae = jnp.abs(preds - targets)
        return (
            jnp.sum((preds - targets) ** 2),
            jnp.sum(ae),
            jnp.sum(ae / jnp.maximum(jnp.abs(targets), 1e-8)),
            preds.size,
        )

    for batch_np in loader:
        batch = {k: jnp.array(v) for k, v in batch_np.items()}
        se, ae, are, count = _eval_batch(params, batch, rng_key)
        total_se += float(se)
        total_ae += float(ae)
        total_are += float(are)
        n += int(count)
    return {
        "rmse_mg_dl": float(np.sqrt(total_se / max(n, 1))),
        "mae_mg_dl": total_ae / max(n, 1),
        "mard_percent": 100.0 * total_are / max(n, 1),
    }


def clear_cuda():
    """Release JAX caches and CUDA memory."""
    jax.clear_caches()
    gc.collect()


def _save_checkpoint(path, params, opt_state, rng, epoch, global_step,
                     best_mae, epochs_without_improvement, history):
    """Atomically save full training state for resume."""
    state = {
        "params": params,
        "opt_state": opt_state,
        "rng": rng,
        "epoch": epoch,
        "global_step": global_step,
        "best_mae": best_mae,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
    }
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)


def _load_checkpoint(path):
    with path.open("rb") as f:
        return pickle.load(f)


def _append_history_row(csv_path, row, write_header):
    """Append one row to the history CSV."""
    keys = list(row.keys())
    with open(csv_path, "a" if not write_header else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def train_single(
    mode,
    data,
    args,
    out_dir,
    lr_override=None,
    structure_builder=create_glucose_transformer,
    evaluate_fn=evaluate,
):
    """Epoch-based training loop for one mode (pc or backprop).

    Validates at the end of each epoch. Early-stops if val MAE does not
    improve for ``patience`` consecutive epochs, or if training becomes
    unstable (NaN/Inf loss).

    Saves a full checkpoint after each epoch for crash recovery.
    Use ``--resume`` to continue from the last checkpoint.

    Returns dict with params, history, best_val_mae, test_metrics, elapsed_s.
    """
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    g_min, g_max = data["g_min"], data["g_max"]

    lr = lr_override if lr_override is not None else args.lr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inference = InferenceSGDNormClip(
        eta_infer=args.eta_infer,
        infer_steps=args.infer_steps,
        max_norm=args.max_infer_norm,
    )
    structure = structure_builder(
        depth=args.depth, embed_dim=args.embed_dim,
        num_heads=args.num_heads, mlp_dim=args.mlp_dim,
        seq_len=args.seq_len, horizon=args.horizon,
        inference=inference, weight_init_std=args.weight_init_std,
        include_output_scaling=args.include_output_scaling,
    )

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(
            initialize_params(structure, jax.random.PRNGKey(0))
        )
    )
    batches_per_epoch = len(train_loader)
    epoch_budget_steps = args.epochs * batches_per_epoch
    max_updates = getattr(args, "max_updates", None)
    total_steps = (
        min(epoch_budget_steps, max_updates)
        if max_updates is not None
        else epoch_budget_steps
    )
    if args.decay_steps is not None and args.decay_epochs is not None:
        raise ValueError("Specify only one of decay_steps and decay_epochs")

    # Try to resume from checkpoint
    ckpt_path = out_dir / "checkpoint.pkl"
    start_epoch = 1
    global_step = 0
    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []
    resumed = False

    warmup = min(args.warmup_steps, total_steps // 2)
    if args.decay_steps is not None:
        decay_steps = args.decay_steps
    elif args.decay_epochs is not None:
        decay_steps = args.decay_epochs * batches_per_epoch
    else:
        decay_steps = total_steps
    if decay_steps <= warmup:
        raise ValueError(
            f"decay_steps ({decay_steps}) must exceed warmup_steps ({warmup})"
        )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(schedule),
    )

    if args.resume and ckpt_path.exists():
        ckpt = _load_checkpoint(ckpt_path)
        params = ckpt["params"]
        opt_state = ckpt["opt_state"]
        rng = ckpt["rng"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_mae = ckpt["best_mae"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt["history"]
        resumed = True
        print(f"Resumed from epoch {ckpt['epoch']}, step {global_step}, "
              f"best val MAE={best_mae:.3f} mg/dL")
    else:
        rng = jax.random.PRNGKey(args.seed)
        rng, init_key = jax.random.split(rng)
        params = initialize_params(structure, init_key)
        opt_state = optimizer.init(params)

    print(
        f"Model: {len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{n_params:,} parameters"
    )
    print(
        f"  {batches_per_epoch} batches/epoch × {args.epochs} epochs; "
        f"budget={total_steps} updates"
    )

    if mode == "pc":
        @jax.jit
        def step_fn(p, o, b, k):
            return train_step(p, o, b, structure, optimizer, k)
    else:
        @jax.jit
        def step_fn(p, o, b, k):
            p, o, loss = train_step_backprop(
                p, o, b, structure, optimizer, k, loss_type="mse"
            )
            return p, o, loss, None

    glucose_range = max(g_max - g_min, 1e-8)

    @jax.jit
    def _batch_metrics(p, b, k):
        """Compute MAE/MARD on a single training batch (glucose scale)."""
        state = compute_forward_pass(p, structure, b, k)
        preds = state.nodes[structure.task_map["y"]].z_mu * glucose_range + g_min
        targets = b["y"] * glucose_range + g_min
        ae = jnp.abs(preds - targets)
        mae = jnp.mean(ae)
        mard = jnp.mean(ae / jnp.maximum(jnp.abs(targets), 1e-8)) * 100.0
        return mae, mard

    mode_label = "PC" if mode == "pc" else "BP"
    metric_label = "energy" if mode == "pc" else "mse"
    print(
        f"\nTraining [{mode_label}] for {args.epochs} epochs "
        f"(starting at {start_epoch}), lr={lr}, patience={args.patience}\n"
    )

    t0 = time.time()
    final_epoch = start_epoch - 1
    unstable = False
    write_header = not resumed or not (out_dir / "history.csv").exists()

    for epoch in range(start_epoch, args.epochs + 1):
        if global_step >= total_steps:
            break
        epoch_loss = 0.0
        epoch_batches = 0
        final_epoch = epoch

        for batch_np in train_loader:
            if global_step >= total_steps:
                break
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
            rng, step_key = jax.random.split(rng)
            params, opt_state, scalar, _ = step_fn(
                params, opt_state, batch, step_key
            )
            scalar_val = float(scalar)
            global_step += 1
            epoch_batches += 1
            epoch_loss += scalar_val

            if not np.isfinite(scalar_val):
                print(
                    f"[{mode_label}] epoch {epoch} batch {epoch_batches}: "
                    f"non-finite {metric_label}={scalar_val} — stopping"
                )
                unstable = True
                break

            if epoch_batches % args.log_every == 0:
                rng, metric_key = jax.random.split(rng)
                mae, mard = _batch_metrics(params, batch, metric_key)
                elapsed = time.time() - t0
                print(
                    f"[{mode_label}] epoch {epoch}  "
                    f"batch {epoch_batches}/{batches_per_epoch}  "
                    f"{metric_label}={scalar_val:.6f}  "
                    f"MAE={float(mae):.2f} mg/dL  "
                    f"MARD={float(mard):.2f}%  "
                    f"elapsed={elapsed:.1f}s"
                )

        if unstable:
            break

        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - t0

        # Validate at end of each epoch
        rng, eval_key = jax.random.split(rng)
        val = evaluate_fn(params, structure, val_loader, eval_key, g_min, g_max)
        row = {
            "epoch": epoch,
            "step": global_step,
            f"avg_{metric_label}": avg_loss,
            **val,
        }
        history.append(row)
        _append_history_row(out_dir / "history.csv", row, write_header)
        write_header = False

        is_best = val["mae_mg_dl"] < best_mae
        if is_best:
            best_mae = val["mae_mg_dl"]
            epochs_without_improvement = 0
            tmp = out_dir / "best_params.pkl.tmp"
            with tmp.open("wb") as f:
                pickle.dump(params, f)
            tmp.replace(out_dir / "best_params.pkl")
        else:
            epochs_without_improvement += 1

        # Save checkpoint for crash recovery
        _save_checkpoint(
            ckpt_path, params, opt_state, rng, epoch, global_step,
            best_mae, epochs_without_improvement, history,
        )

        best_tag = " ★" if is_best else ""
        print(
            f"  [VAL] epoch {epoch}/{args.epochs}  "
            f"mae={val['mae_mg_dl']:.3f} mg/dL  "
            f"rmse={val['rmse_mg_dl']:.3f}  "
            f"mard={val['mard_percent']:.2f}%  "
            f"avg_{metric_label}={avg_loss:.6f}  "
            f"elapsed={elapsed:.1f}s{best_tag}"
        )

        if best_mae < float("inf") and val["mae_mg_dl"] > 2.0 * best_mae:
            print(
                f"  Divergence guard: val MAE {val['mae_mg_dl']:.1f} > "
                f"2× best {best_mae:.1f} — stopping"
            )
            break

        if global_step >= total_steps:
            print(f"  Update budget reached: {global_step}/{total_steps}")
            break

        if epochs_without_improvement >= args.patience and epoch < args.epochs:
            print(
                f"  Early stop: {args.patience} epochs without improvement"
            )
            break

    elapsed = time.time() - t0

    # Test evaluation on held-out set using best checkpoint
    test_metrics = None
    best_path = out_dir / "best_params.pkl"
    if best_path.exists():
        with best_path.open("rb") as f:
            best_params = pickle.load(f)
        rng, test_key = jax.random.split(rng)
        test_metrics = evaluate_fn(
            best_params, structure, test_loader, test_key, g_min, g_max
        )
        print(
            f"\n  [TEST] mae={test_metrics['mae_mg_dl']:.3f} mg/dL  "
            f"rmse={test_metrics['rmse_mg_dl']:.3f}  "
            f"mard={test_metrics['mard_percent']:.2f}%"
        )

    print(f"\nTraining completed in {elapsed:.1f}s ({final_epoch} epochs)")
    if best_mae < float("inf"):
        print(f"Best val MAE: {best_mae:.3f} mg/dL")

    # Write config
    meta = {
        "mode": mode,
        "depth": args.depth,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "mlp_dim": args.mlp_dim,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "final_epoch": final_epoch,
        "total_steps": global_step,
        "planned_updates": total_steps,
        "decay_steps": decay_steps,
        "decay_epochs": args.decay_epochs,
        "lr": lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "eta_infer": args.eta_infer,
        "infer_steps": args.infer_steps,
        "max_infer_norm": args.max_infer_norm,
        "grad_clip": args.grad_clip,
        "weight_init_std": args.weight_init_std,
        "include_output_scaling": args.include_output_scaling,
        "best_val_mae_mg_dl": best_mae,
        "g_min": g_min,
        "g_max": g_max,
        "n_params": n_params,
        "elapsed_s": round(elapsed, 1),
        "unstable": unstable,
    }
    if test_metrics:
        meta["test_mae_mg_dl"] = test_metrics["mae_mg_dl"]
        meta["test_rmse_mg_dl"] = test_metrics["rmse_mg_dl"]
        meta["test_mard_percent"] = test_metrics["mard_percent"]
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    return {
        "params": params,
        "history": history,
        "best_val_mae": best_mae,
        "test_metrics": test_metrics,
        "elapsed_s": elapsed,
        "config": meta,
    }


def compare(args):
    """Run PC then backprop on the same data/seed, clean CUDA between, write comparison."""
    data = prepare_data(
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()}, devices: {devices}")
    print(
        f"Data: {data['n_train']} train, {data['n_val']} val, "
        f"{data['n_test']} test windows  "
        f"(seq_len={args.seq_len}, horizon={args.horizon})"
    )

    base_dir = Path(args.out_dir)

    # --- PC ---
    print("\n" + "=" * 60)
    print("PHASE 1: Predictive Coding")
    print("=" * 60)
    pc_result = train_single(
        "pc", data, args, out_dir=base_dir / "pc", lr_override=args.lr,
    )
    del pc_result["params"]
    clear_cuda()
    print("\nCUDA memory released.\n")

    # --- Backprop ---
    print("=" * 60)
    print("PHASE 2: Backpropagation")
    print("=" * 60)
    bp_result = train_single(
        "backprop", data, args, out_dir=base_dir / "backprop",
        lr_override=args.lr_backprop,
    )
    del bp_result["params"]
    clear_cuda()

    # --- Comparison summary ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    summary = {
        "architecture": {
            "depth": args.depth,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "mlp_dim": args.mlp_dim,
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "n_params": pc_result["config"]["n_params"],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "warmup_steps": args.warmup_steps,
            "patience": args.patience,
        },
        "pc": {
            "lr": args.lr,
            "eta_infer": args.eta_infer,
            "infer_steps": args.infer_steps,
            "final_epoch": pc_result["config"]["final_epoch"],
            "total_steps": pc_result["config"]["total_steps"],
            "best_val_mae_mg_dl": pc_result["best_val_mae"],
            "unstable": pc_result["config"]["unstable"],
            "elapsed_s": pc_result["elapsed_s"],
        },
        "backprop": {
            "lr": args.lr_backprop,
            "final_epoch": bp_result["config"]["final_epoch"],
            "total_steps": bp_result["config"]["total_steps"],
            "best_val_mae_mg_dl": bp_result["best_val_mae"],
            "unstable": bp_result["config"]["unstable"],
            "elapsed_s": bp_result["elapsed_s"],
        },
    }
    if pc_result["test_metrics"]:
        summary["pc"]["test_mae_mg_dl"] = pc_result["test_metrics"]["mae_mg_dl"]
        summary["pc"]["test_rmse_mg_dl"] = pc_result["test_metrics"]["rmse_mg_dl"]
        summary["pc"]["test_mard_percent"] = pc_result["test_metrics"]["mard_percent"]
    if bp_result["test_metrics"]:
        summary["backprop"]["test_mae_mg_dl"] = bp_result["test_metrics"]["mae_mg_dl"]
        summary["backprop"]["test_rmse_mg_dl"] = bp_result["test_metrics"]["rmse_mg_dl"]
        summary["backprop"]["test_mard_percent"] = bp_result["test_metrics"]["mard_percent"]

    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "comparison.json").write_text(json.dumps(summary, indent=2))

    pc_mae = summary["pc"].get("test_mae_mg_dl", summary["pc"]["best_val_mae_mg_dl"])
    bp_mae = summary["backprop"].get(
        "test_mae_mg_dl", summary["backprop"]["best_val_mae_mg_dl"]
    )
    metric_source = "test" if "test_mae_mg_dl" in summary["pc"] else "val"
    winner = "PC" if pc_mae < bp_mae else "Backprop"

    lines = [
        f"  PC       {metric_source} MAE: {pc_mae:.3f} mg/dL  "
        f"(epochs={summary['pc']['final_epoch']}, {summary['pc']['elapsed_s']:.0f}s)",
        f"  Backprop {metric_source} MAE: {bp_mae:.3f} mg/dL  "
        f"(epochs={summary['backprop']['final_epoch']}, {summary['backprop']['elapsed_s']:.0f}s)",
        f"  Winner: {winner} (Δ = {abs(pc_mae - bp_mae):.3f} mg/dL)",
    ]
    for line in lines:
        print(line)

    with open(base_dir / "comparison.txt", "w") as f:
        f.write("Glucose Transformer — PC vs Backprop Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Architecture: depth={args.depth}, d={args.embed_dim}, "
                f"heads={args.num_heads}, ff={args.mlp_dim}\n")
        f.write(f"Data: seq_len={args.seq_len}, horizon={args.horizon}, "
                f"seed={args.seed}\n")
        f.write(f"Budget: {args.epochs} epochs, batch_size={args.batch_size}, "
                f"patience={args.patience}\n\n")
        f.write("Results:\n")
        for line in lines:
            f.write(line + "\n")
        f.write(f"\nFull metrics: {base_dir / 'comparison.json'}\n")

    print(f"\nResults saved to {base_dir}/")
    return summary


def main():
    args = parse_args()
    if args.mode == "compare":
        compare(args)
    else:
        data = prepare_data(
            seq_len=args.seq_len,
            horizon=args.horizon,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        devices = jax.devices()
        print(f"JAX backend: {jax.default_backend()}, devices: {devices}")
        print(
            f"Data: {data['n_train']} train, {data['n_val']} val, "
            f"{data['n_test']} test windows  "
            f"(seq_len={args.seq_len}, horizon={args.horizon})"
        )
        lr = args.lr if args.mode == "pc" else args.lr_backprop
        train_single(args.mode, data, args, out_dir=args.out_dir, lr_override=lr)


if __name__ == "__main__":
    main()
