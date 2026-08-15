"""Compare DAG ePC with the existing clipped-sPC ResNet-18 baseline.

Modes:
    paired       Train paired ePC/sPC arms and report accuracy/time statistics.
    convergence  Compare inference energy trajectories from identical parameters
                 and initial state, excluding JIT compilation from step timing.

Examples:
    python examples/epc_spc_resnet18_compare.py --mode convergence
    python examples/epc_spc_resnet18_compare.py --mode paired --n_trials 3

The paired benchmark is intentionally a manual GPU experiment. Results vary by
hardware and JAX version; record the environment alongside any result table.

Reference run (2026-08-14, RTX 4090, JAX 0.6.2, CUDA 12 plugin 0.6.2):

===========================  ===================  =====================
Metric                       ePC-10               sPC-clip-120
===========================  ===================  =====================
Accuracy, n=3 x 2 epochs     28.84 +/- 0.76%      34.24 +/- 0.92%
Training seconds/epoch       44.357 +/- 0.748     295.150 +/- 0.900
Step-120 energy              3.9824901            5.4600258
===========================  ===================  =====================

ePC reached the sPC terminal energy at step 2 and trained 6.65x faster, but
the preregistered strict gate failed because mean accuracy was 5.39 percentage
points lower. Full protocol and trace:
``docs/benchmark_results/epc_resnet18/2026-08-14.md``.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import importlib.util
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core import EPCInference, InferenceSGDNormClip
from fabricpc.experiments import ABExperiment, ExperimentArm
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_history
from fabricpc.utils.data.dataloader import Cifar10Loader

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "resnet18_cifar10_demo.py")
_SPEC = importlib.util.spec_from_file_location("resnet18_cifar10_demo", _DEMO_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load ResNet demo from {_DEMO_PATH}")
_RESNET_DEMO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESNET_DEMO)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paired", "convergence"), default="paired")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--spc_steps", type=int, default=120)
    parser.add_argument("--spc_eta", type=float, default=0.1)
    parser.add_argument("--epc_steps", type=int, default=10)
    parser.add_argument("--epc_eta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--track_steps", type=int, default=120)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _epc(args, *, steps=None):
    return EPCInference(
        eta_infer=args.epc_eta,
        infer_steps=args.epc_steps if steps is None else steps,
    )


def _spc(args, *, steps=None):
    return InferenceSGDNormClip(
        eta_infer=args.spc_eta,
        infer_steps=args.spc_steps if steps is None else steps,
        max_norm=1.0,
    )


def _model_factory(inference):
    def factory(rng_key):
        return _RESNET_DEMO._create_mupc_model(
            rng_key,
            inference=inference,
            activation=_RESNET_DEMO.get_activation("relu"),
        )

    return factory


def _loader_factory(args):
    def factory(seed):
        return (
            Cifar10Loader("train", batch_size=args.batch_size, shuffle=True, seed=seed),
            Cifar10Loader("test", batch_size=args.batch_size, shuffle=False, seed=seed),
        )

    return factory


def _optimizer(args, steps_per_epoch):
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    return optax.adamw(schedule, weight_decay=args.weight_decay)


def run_paired(args):
    loader_factory = _loader_factory(args)
    prototype_train, _ = loader_factory(0)
    steps_per_epoch = len(prototype_train)
    train_config = {"num_epochs": args.num_epochs}

    epc_arm = ExperimentArm(
        name=f"ePC-{args.epc_steps}",
        model_factory=_model_factory(_epc(args)),
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=_optimizer(args, steps_per_epoch),
        train_config=train_config,
    )
    spc_arm = ExperimentArm(
        name=f"sPC-clip-{args.spc_steps}",
        model_factory=_model_factory(_spc(args)),
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=_optimizer(args, steps_per_epoch),
        train_config=train_config,
    )
    experiment = ABExperiment(
        arm_a=epc_arm,
        arm_b=spc_arm,
        metric="accuracy",
        data_loader_factory=loader_factory,
        n_trials=args.n_trials,
        verbose=args.verbose,
    )
    experiment.run().print_summary()


def _energy_curve(metrics, structure):
    curve = None
    for name, node in structure.nodes.items():
        if node.node_info.in_degree == 0:
            continue
        node_energy = np.asarray(metrics[name]["energy"])
        curve = node_energy if curve is None else curve + node_energy
    if curve is None:
        return np.zeros((0,), dtype=np.float32)
    return curve


def _time_solver_steps(solver, params, state, clamps, structure, step_count):
    solver_cls = type(solver)
    state = solver_cls.begin_segment(params, state, clamps, structure)
    compiled_step = jax.jit(
        lambda current_state: solver_cls.inference_step(
            params, current_state, clamps, structure, solver.config
        )
    )
    jax.block_until_ready(compiled_step(state))

    start = time.perf_counter()
    for _ in range(step_count):
        state = compiled_step(state)
    jax.block_until_ready(state)
    return (time.perf_counter() - start) / step_count


def run_convergence(args):
    if args.track_steps < 1:
        raise ValueError("--track_steps must be >= 1")

    epc_solver = _epc(args, steps=args.track_steps)
    spc_solver = _spc(args, steps=args.track_steps)
    params, epc_structure = _model_factory(epc_solver)(jax.random.PRNGKey(42))
    spc_structure = epc_structure._replace(
        config={**epc_structure.config, "inference": spc_solver}
    )

    test_loader = Cifar10Loader(
        "test", batch_size=args.batch_size, shuffle=False, seed=42
    )
    images, targets = next(iter(test_loader))
    clamps = {"input": jnp.asarray(images), "output": jnp.asarray(targets)}
    initial_state = initialize_graph_state(
        epc_structure,
        images.shape[0],
        jax.random.PRNGKey(43),
        clamps=clamps,
        params=params,
    )

    _, epc_metrics = run_inference_with_history(
        params, initial_state, clamps, epc_structure
    )
    _, spc_metrics = run_inference_with_history(
        params, initial_state, clamps, spc_structure
    )
    epc_curve = _energy_curve(epc_metrics, epc_structure)
    spc_curve = _energy_curve(spc_metrics, spc_structure)
    target_energy = spc_curve[-1]
    reached = np.flatnonzero(epc_curve <= target_energy)

    epc_seconds = _time_solver_steps(
        epc_solver,
        params,
        initial_state,
        clamps,
        epc_structure,
        args.track_steps,
    )
    spc_seconds = _time_solver_steps(
        spc_solver,
        params,
        initial_state,
        clamps,
        spc_structure,
        args.track_steps,
    )

    print("step,ePC_energy,sPC_energy")
    for step, (epc_energy, spc_energy) in enumerate(zip(epc_curve, spc_curve), start=1):
        print(f"{step},{epc_energy:.8g},{spc_energy:.8g}")
    print()
    print(f"sPC final energy E*: {target_energy:.8g}")
    if reached.size:
        print(f"ePC first reaches E*: step {int(reached[0]) + 1}")
    else:
        print(f"ePC did not reach E* within {args.track_steps} steps")
    print(f"ePC post-warmup seconds/step: {epc_seconds:.6f}")
    print(f"sPC post-warmup seconds/step: {spc_seconds:.6f}")
    print(f"per-step time ratio (ePC/sPC): {epc_seconds / spc_seconds:.3f}")


def main():
    args = parse_args()
    if args.mode == "paired":
        run_paired(args)
    else:
        run_convergence(args)


if __name__ == "__main__":
    main()
