"""Timing and memory measurement for one benchmark trial.

Two rules keep the numbers honest:

* The first step includes JIT compilation, so it is timed on its own and
  never mixed into the per-step time.
* JAX hands work to the device and returns right away, so every timed step
  ends with ``block_until_ready``. Without that the stopwatch measures
  nothing.
"""

import itertools
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Optional

import jax

from fabricpc.core.epsilon_spectrum import make_epsilon_spectrum
from fabricpc.core.inference_epc import EPCInference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training import make_train_step
from fabricpc.training.trainer import build_clamps, convert_batch


@dataclass(frozen=True)
class Timing:
    compile_time_s: float  # lowering and compiling the step, nothing else
    step_time_ms: float  # median over the timed steps
    timed_steps: int
    step_memory: Optional[Dict[str, int]] = None  # see step_memory()


def _compile(step, params, opt_state, batch, key):
    """Compile ``step`` once; return (callable, compiled-or-None, seconds).

    A jitted step is lowered and compiled explicitly, so the compile is
    timed on its own and the same executable serves the timing and the
    memory report. Anything else is used as it is.
    """
    lower = getattr(step, "lower", None)
    if lower is None:
        return step, None, 0.0
    t0 = time.perf_counter()
    compiled = lower(params, opt_state, batch, key).compile()
    return compiled, compiled, time.perf_counter() - t0


def time_steps(
    params,
    structure,
    optimizer,
    loader,
    rng_key,
    *,
    algorithm: str,
    warmup_steps: int,
    timed_steps: int,
) -> Timing:
    """Time one training step, the way a benchmark should.

    Compiles the step once (timed on its own), runs ``warmup_steps`` untimed
    steps plus one more to absorb any first-run cost, then ``timed_steps``
    timed steps. Only the batches needed are read from ``loader``, and they
    are reused in a cycle if the loader is shorter. The compiled step's
    memory needs come back in ``Timing.step_memory``.
    """
    step = make_train_step(structure, optimizer, algorithm=algorithm)
    opt_state = optimizer.init(params)
    needed = 1 + warmup_steps + timed_steps
    batches = [convert_batch(b) for b in itertools.islice(loader, needed)]
    if not batches:
        raise ValueError("loader yielded no batches")

    def batch_at(i):
        return batches[i % len(batches)]

    keys = jax.random.split(rng_key, 1 + warmup_steps + timed_steps)
    run, compiled, compile_time_s = _compile(
        step, params, opt_state, batch_at(0), keys[0]
    )

    # Warmup, including the first call: run, do not time.
    for i in range(1 + warmup_steps):
        params, opt_state, _, _ = run(params, opt_state, batch_at(i), keys[i])
    jax.block_until_ready(params)

    # Timed steps, each one synced before the clock stops.
    samples = []
    for i in range(timed_steps):
        j = 1 + warmup_steps + i
        t0 = time.perf_counter()
        params, opt_state, _, _ = run(
            params, opt_state, batch_at(j), keys[j % len(keys)]
        )
        jax.block_until_ready(params)
        samples.append(time.perf_counter() - t0)

    return Timing(
        compile_time_s=compile_time_s,
        step_time_ms=statistics.median(samples) * 1000.0,
        timed_steps=timed_steps,
        step_memory=_memory_of(compiled),
    )


@dataclass(frozen=True)
class MemorySnapshot:
    bytes_in_use: Optional[int]  # what is held right now
    peak_bytes: Optional[int]  # the most ever held so far in this process


def memory_snapshot() -> MemorySnapshot:
    """Read the default device's memory numbers, or None where it cannot say.

    GPU devices report both. The CPU device reports nothing, so on a laptop
    both are None and the result file says so instead of guessing.

    The peak counts everything since the process started, compile buffers
    included. Each trial normally gets its own process, so the peak is that
    trial's alone; with ``--in-process`` a later trial's peak can carry over
    from an earlier, bigger one.
    """
    jax.block_until_ready(jax.numpy.zeros(1))
    device = jax.local_devices()[0]
    stats = device.memory_stats() if hasattr(device, "memory_stats") else None
    if not stats:
        return MemorySnapshot(bytes_in_use=None, peak_bytes=None)

    def read(key):
        return int(stats[key]) if key in stats else None

    return MemorySnapshot(
        bytes_in_use=read("bytes_in_use"), peak_bytes=read("peak_bytes_in_use")
    )


def _memory_of(compiled) -> Optional[Dict[str, int]]:
    """Argument, output and temporary bytes of a compiled step, from XLA."""
    if compiled is None:
        return None
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, NotImplementedError):
        return None
    if analysis is None:
        return None
    sizes = {
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "temp_bytes": int(analysis.temp_size_in_bytes),
        "alias_bytes": int(analysis.alias_size_in_bytes),
    }
    sizes["total_bytes"] = (
        sizes["argument_bytes"]
        + sizes["output_bytes"]
        + sizes["temp_bytes"]
        - sizes["alias_bytes"]
    )
    return sizes


def step_memory(
    params, structure, optimizer, loader, rng_key, *, algorithm: str
) -> Optional[Dict[str, int]]:
    """Memory one compiled training step needs, as reported by XLA.

    The process-wide peak from ``memory_snapshot`` also counts scratch space
    used while compiling (on GPU, cuDNN tries several convolution algorithms
    in temporary buffers), and that depends only on the layer shapes, so it
    comes out the same for backprop, sPC and ePC. The compiled step's own
    sizes are what actually differ: its inputs (params, optimizer state,
    batch), its outputs, and the temporary buffers it needs while running.

    ``time_steps`` already reports this from the step it compiled; this
    function compiles one on its own. Returns None when the step cannot be
    analysed (for example a step that is not jitted).
    """
    step = make_train_step(structure, optimizer, algorithm=algorithm)
    batch = convert_batch(next(iter(loader)))
    _, compiled, _ = _compile(step, params, optimizer.init(params), batch, rng_key)
    return _memory_of(compiled)


def epc_regime(params, structure, batch, rng_key, *, iters: int = 30):
    """How PC-like an ePC graph's relaxation is, on one batch; None if not ePC.

    With a small rate or few steps an ePC run is backprop-like: one step from
    zero error is backprop's activation gradient. ``EPCInference.regime``
    reads that off the error-Hessian spectrum (see the Training with ePC
    guide). The band is "backprop-like" when the gradient-weighted relaxed
    fraction f_weighted is below 0.1, "near PC equilibrium" above 0.9, and
    "partially relaxed" between.
    """
    inference = structure.config.get("inference")
    if not isinstance(inference, EPCInference):
        return None
    clamps = build_clamps(convert_batch(batch), structure, clamp_target=True)
    batch_size = next(iter(clamps.values())).shape[0]
    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    spectrum = make_epsilon_spectrum(structure, iters)(
        params, state, clamps, rng_key
    ).host()
    r = inference.regime(spectrum)
    return {
        "band": r.band,
        "f_weighted": float(r.f_weighted),
        "f_max": float(r.f_max),
        "unstable": bool(r.unstable),
        "eta_lambda_max": float(r.eta_lambda_max),
        "lambda_max": float(r.lambda_max),
        "lambda_min": float(r.lambda_min),
        "label": str(r),
    }
