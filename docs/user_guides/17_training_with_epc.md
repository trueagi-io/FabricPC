# Training with ePC

This guide is for a researcher who has run the [Quickstart](02_quickstart.md) and wants to train their own model with the error-based solver `EPCInference` and report the result correctly. The rate and step count are the user's choice; the library measures what the choice depends on. Every number below comes from the guide's own runnable example, a linear chain trained to imitate a teacher whose gain forces the weights to grow; on such a graph the theory behind the flags is exact, so what the flags predict is what happens. Your graph gives different numbers, and the procedure derives them from your graph. The measurements on a deep network, the ResNet-18 demo, are in the report `docs/reports/epc_regime_and_stability_report.md` and are pointed to by section name.

## What ePC changes

The state-based solvers relax the latent states one node at a time, so the output clamp reaches an early layer only after as many steps as there are hops. `EPCInference` relaxes the prediction errors ε instead. The latents are derived from the errors by one forward pass along the graph's schedule (`z_latent = z_mu + ε`), and one reverse pass per step through the whole derived forward delivers the loss signal to every layer at once, so a few steps replace the hundreds a state-based solver needs on a deep DAG, at backprop-scale memory per step. The two parameterizations share the same energy and the same equilibria on DAGs. On cyclic graphs, built with `graph(..., unroll=U)`, ePC minimizes the unrolled approximation of the energy at degree U while the state-based solvers minimize the exact graph energy. The update rule and the segment hooks are in the [Inference Algorithms API](12_api_inference.md#epcinference).

## Three things to know before the first run

**The defaults have no regime meaning independent of your graph.** `EPCInference()` with no arguments is rate 1e-3 and 5 steps. On the ResNet-18 demo that pair was backprop-like at init and the run collapsed over the full schedule; on another graph the same pair may sit in any band or above the stability bound. Pass both arguments, measure the spectrum of your graph, and read the regime label before training.

**The rate is one global rate, bounded by your graph at your init.** A change in one node's error moves every downstream derived latent, so `eta_infer` acts through the whole network and is tuned like a weight learning rate, not like a state-based solver's per-node rate. Gradient descent on the errors is stable only below 2/λ_max, where λ_max is the largest excited eigenvalue of the energy Hessian in error coordinates, and at odd step counts the output layer is damaged before that bound. λ_max spans orders of magnitude across graphs and initializations, so the state-based ranges elsewhere in the guides, and any rate carried from another graph, are not rates for yours.

**λ_max grows during training, and a fixed rate has no lasting margin.** On the demo the rate two orders of magnitude below the init bound still crossed it after a few epochs, and the run fell to chance (report Section 6.2, Implications for FabricPC). Attach `RegimeProbe` to every run. The library detects the crossing; it does not yet control the rate.

## Symbols

| Symbol | Meaning | Where it lives |
|---|---|---|
| η | `eta_infer`, the rate on the errors | `EPCInference` |
| T | `infer_steps`, the number of error updates per weight update | `EPCInference` |
| H_ε | the Hessian of the total energy in error coordinates at the feedforward state of one batch | measured by `epsilon_spectrum` |
| λ_max, λ_min | the largest and smallest eigenvalues of H_ε among the modes the starting gradient excites | `EpsilonSpectrum.lambda_max`, `.lambda_min` |
| g₀ | the gradient of the energy with respect to the errors at ε = 0, the direction one step moves | inside `epsilon_spectrum` |
| w_k | the fraction of ‖g₀‖² carried by mode k | `EpsilonSpectrum.ritz_weights` |
| f(λ) | 1 − (1 − ηλ)^T, how far a mode of eigenvalue λ has relaxed toward its equilibrium after T steps | formula |
| f̄ | Σ_k w_k f(λ_k) over the positive-curvature modes, the gradient-weighted relaxed fraction | `Regime.f_weighted` |
| f_max | f(λ_max), the relaxed fraction of the top mode | `Regime.f_max` |
| `negative_weight` | the gradient weight on negative-curvature modes | `EpsilonSpectrum`, `Regime` |
| `growth_min` | (1 + η·max(0, −λ_min))^T, the T-step growth of the most negative mode | `Regime` only |

The band is read on f̄: below 0.1 backprop-like, above 0.9 near PC equilibrium, between them partially relaxed. At the equilibrium the output error is re-weighted by the network's transfer function; the report's Section 5.10 (The equilibrium damps the learning signal by S⁻¹) derives this on a linear chain.

## The workflow

The blocks below run on a synthetic example chosen so that every flag's prediction is exact and the failure it predicts is visible within seconds on a CPU: a chain of linear layers with Gaussian energies, trained to imitate a linear teacher whose gain (largest singular value) is far above the student's gain at init, so the fit has to grow the weights and λ_max with them. On a linear-Gaussian graph the local quadratic model behind the flags is the whole energy (report Section 2.2, Why a linear network gives an exact answer). Replace the chain with your graph and the teacher batches with your loader; every substitution point is marked. A note on nonlinear models follows Step 6.

### Step 1: Build the graph with an explicit solver

Build with a provisional pair; Step 4 replaces it.

```python
import jax
import jax.numpy as jnp
import optax
from fabricpc import setup_jax
from fabricpc.core import EPCInference
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, Linear

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

D_IN, WIDTH, D_OUT, BATCH = 16, 32, 8, 64


def build(inference, hidden=2):
    """A chain of linear layers with Gaussian energies: your graph goes here."""
    x = IdentityNode(shape=(D_IN,), name="x")
    nodes, edges, prev = [x], [], x
    for i in range(hidden):
        h = Linear(shape=(WIDTH,), activation=IdentityActivation(), name=f"h{i + 1}",
                   weight_init=XavierInitializer())
        edges.append(Edge(source=prev, target=h.slot("in")))
        nodes.append(h)
        prev = h
    y = Linear(shape=(D_OUT,), activation=IdentityActivation(), energy=GaussianEnergy(),
               name="y", weight_init=XavierInitializer())
    edges.append(Edge(source=prev, target=y.slot("in")))
    nodes.append(y)
    return graph(nodes=nodes, edges=edges, task_map=TaskMap(x=x, y=y), inference=inference)


def teacher(key, gain):
    """A linear map with largest singular value ``gain``."""
    A = jax.random.normal(key, (D_OUT, D_IN))
    return gain * A / jnp.linalg.norm(A, 2)


def teacher_batches(A, key, n_batches):
    """``n_batches`` dict batches of (x, A x): your data loader goes here."""
    out = []
    for i in range(n_batches):
        x = jax.random.normal(jax.random.fold_in(key, i), (BATCH, D_IN))
        out.append({"x": x, "y": x @ A.T})
    return out


graph_key, train_key, data_key, probe_key = jax.random.split(jax.random.PRNGKey(0), 4)
A = teacher(jax.random.fold_in(data_key, 0), gain=20.0)
train_batches = teacher_batches(A, jax.random.fold_in(data_key, 1), 60)
test_batches = teacher_batches(A, jax.random.fold_in(data_key, 2), 10)

structure = build(EPCInference(eta_infer=0.01, infer_steps=4))  # provisional pair
params = initialize_params(structure, graph_key)
```

### Step 2: Measure the spectrum at init

The spectrum is measured at the feedforward state of one clamped batch. `build_clamps` turns a batch dict into the node clamps, `initialize_graph_state` runs the graph's state initializer, and `epsilon_spectrum` runs Lanczos on Hessian-vector products through the solver's error energy.

```python
from fabricpc.core import epsilon_spectrum
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.training import build_clamps

clamps_a = build_clamps(test_batches[0], structure, clamp_target=True)
clamps_b = build_clamps(test_batches[1], structure, clamp_target=True)


def spectrum_at(structure, params, clamps, key=jax.random.PRNGKey(1)):
    state = initialize_graph_state(structure, BATCH, key, clamps=clamps, params=params)
    return epsilon_spectrum(params, state, clamps, structure)


for label, clamps in [("batch A", clamps_a), ("batch B", clamps_b)]:
    s = spectrum_at(structure, params, clamps)
    print(f"{label}: lambda_max = {float(s.lambda_max):.4g}, bound 2/lambda_max = {2 / float(s.lambda_max):.4g}, "
          f"lambda_min = {float(s.lambda_min):.4g}, negative_weight = {float(s.negative_weight):.3g}")
spectrum = spectrum_at(structure, params, clamps_a)
```

Output at the time of writing:

```text
batch A: lambda_max = 9.416, bound 2/lambda_max = 0.2124, lambda_min = 1.773, negative_weight = 0
batch B: lambda_max = 9.416, bound 2/lambda_max = 0.2124, lambda_min = 1.773, negative_weight = 0
```

How to read it. On a linear graph H_ε depends on the weights alone, so the two batches give the same spectrum; on a nonlinear graph the activation derivatives enter and the batches differ (on the quickstart's sigmoid perceptron by about one percent), so measure a second batch and take the spread as the uncertainty of the bound. Under the default `FeedforwardStateInit` the key does not change the state, so it does not change the spectrum; a random state initializer makes the key matter, and then a second key belongs in the spread. λ_min from 30 Lanczos steps is a sign and an order of magnitude, not a converged value; a negative λ_min with nonzero `negative_weight` means the energy is indefinite at this state, which nonlinear graphs reach as the weights grow (report Section 6.3, Limits). The value that decides the run is not this one but its growth during training (Step 6).

### Step 3: Tabulate the regime for a grid

`EPCInference(eta, T).regime(spectrum)` is a formula on the measured spectrum and costs nothing, so tabulate the band for a grid before training. Express η as a fraction of the measured bound 2/λ_max, so the table transfers across graphs; the grid below spans fractions from one to thirty percent of the bound and step counts from one to twenty.

```python
fractions = [0.01, 0.03, 0.1, 0.3]
steps = [1, 2, 4, 5, 10, 20]
lam_max = float(spectrum.lambda_max)
print(f"{'eta':>9} {'fraction of bound':>18} " + " ".join(f"{'T=' + str(T):>26}" for T in steps))
for c in fractions:
    eta = c * 2.0 / lam_max
    cells = []
    for T in steps:
        r = EPCInference(eta_infer=eta, infer_steps=T).regime(spectrum)
        flag = "!" if r.unstable else ("r" if r.output_gradient_reverses else " ")
        cells.append(f"{r.band:>19} {r.f_weighted:4.2f}{flag}")
    print(f"{eta:9.4g} {c:18.2f} " + " ".join(f"{cell:>26}" for cell in cells))
```

Output at the time of writing (band and f̄; `!` marks `unstable`, `r` marks `output_gradient_reverses`):

```text
      eta  fraction of bound                        T=1                        T=2                        T=4                        T=5                       T=10                       T=20
 0.002124               0.01        backprop-like 0.01         backprop-like 0.03         backprop-like 0.05         backprop-like 0.06     partially relaxed 0.12     partially relaxed 0.22 
 0.006372               0.03        backprop-like 0.04         backprop-like 0.07     partially relaxed 0.14     partially relaxed 0.17     partially relaxed 0.31     partially relaxed 0.51 
  0.02124               0.10    partially relaxed 0.13     partially relaxed 0.23     partially relaxed 0.40     partially relaxed 0.47     partially relaxed 0.69     partially relaxed 0.87 
  0.06372               0.30    partially relaxed 0.38     partially relaxed 0.59     partially relaxed 0.79     partially relaxed 0.84   near PC equilibrium 0.95   near PC equilibrium 0.99 
```

Your graph's table is your range. Rows further down have less margin to the bound; columns further right cost more reverse passes per weight update.

### Step 4: Choose the solver and (η, T)

Two design choices are made here: a plain `EPCInference`, or an `InferenceSchedule` that runs a few ePC steps and then a state-based segment ([Composing ePC and sPC](#composing-epc-and-spc)); and the cell of the table to train in.

The three bands and what they cost:

- **Backprop-like** (f̄ < 0.1): the weight gradients are backprop's with the hidden layers scaled by η, and Adam normalizes the scale away. Fastest per update, and a result in this band is a backprop result through a slower path.
- **Near PC equilibrium** (f̄ > 0.9): the errors have settled. Reaching it needs η·T·λ ≳ 3 on the modes that carry the gradient, so T grows with the ratio of λ_max to the slowest gradient-carrying eigenvalue. At the equilibrium the output error is re-weighted by the network's transfer function, which a per-parameter optimizer cannot undo; on the demo the equilibrium cells reached a lower accuracy plateau than the backprop-like cells, and the report presents the linear-chain derivation of this as a consistency argument, not a measurement (Section 5.10).
- **Partially relaxed**: between the two. The label reports where.

Read both f̄ and f_max. The band is defined on f̄, the energy criterion; on the demo the accuracy transition at init followed f_max instead, and which quantity should label the backprop-like edge is open (report Section 6.4, Follow-up). `Regime` carries both.

On η: start well below the bound, because growth during training, not the margin at init, decides the run. The odd-T reversal bound η·(λ_max − 1) < 1 is tighter than the stability bound at T = 1. This guide gives no fixed numeric range for η or T; the table is the range.

The example takes the cell at 0.3 of the bound with T = 4. An even T is chosen so that the only flag in play is the stability crossing; Step 7 shows what an odd T adds.

```python
eta_chosen, T_chosen = round(0.3 * 2.0 / lam_max, 4), 4
structure = build(EPCInference(eta_infer=eta_chosen, infer_steps=T_chosen))
params = initialize_params(structure, graph_key)
print(EPCInference(eta_infer=0.01 * 2.0 / lam_max, infer_steps=1).regime(spectrum))
print(structure.config["inference"].regime(spectrum))
```

```text
eta*T*lambda_max = 0.02 (gradient-weighted relaxed fraction 0.01, fastest mode 0.02): backprop-like
eta*T*lambda_max = 2.4 (gradient-weighted relaxed fraction 0.79, fastest mode 0.97): partially relaxed
```

### Step 5: Attach the probe

`RegimeProbe` re-measures the spectrum every `every` weight updates on a fixed probe batch, records the regime flags of the configured solver and the Frobenius norm of every weight, and dates each row by update and epoch; `on_epoch` adds the epoch's test accuracy so a collapse can be dated against the spectrum. `every` and `key` are required keyword arguments; `iters=30` is the Lanczos budget per probe. Any `iter_callback` forces a device sync per batch; the probe itself costs one feedforward initialization and `iters` Hessian-vector products per probe.

The iteration callback below also keeps a copy of the parameters and optimizer state at the most recent probe that raised no flag, so a remedy can restart from a state the flags cleared. The trainer donates its parameter buffers, so the copy is required ([Callbacks](08_training_and_evaluation.md#callbacks)). `evaluate` returns a metrics dict; a classifier passes its `"accuracy"` entry to `on_epoch`, and this regression example passes `None` and prints the test energy.

```python
from fabricpc.training import RegimeProbe, evaluate, train

probe = RegimeProbe(structure, clamps_a, every=10, key=probe_key, csv_path="epc_regime_track.csv")
last_clear = {}  # the most recent probed state with no flag raised


def on_iter(ctx):
    probe.on_iter(ctx)
    if ctx.step % probe.every == 0:
        row = probe.probe_rows()[-1]
        if not row["unstable"] and not row["output_gradient_reverses"]:
            last_clear.update(
                step=ctx.step, epoch=ctx.epoch_idx,
                params=jax.tree_util.tree_map(jnp.copy, ctx.params),
                opt_state=jax.tree_util.tree_map(jnp.copy, ctx.opt_state),
            )


def on_epoch(ctx):
    metrics = evaluate(ctx.params, ctx.structure, test_batches, ctx.config, ctx.epoch_key)
    probe.on_epoch(ctx, None)  # a classifier passes metrics["accuracy"] here
    print(f"epoch {ctx.epoch_idx + 1}: test energy {float(metrics['target_energy']):.4g}, "
          f"train energy {float(ctx.metrics['energy']):.4g}")


result = train(params, structure, train_batches, optax.adam(3e-3), {"num_epochs": 4}, train_key,
               verbose=False, iter_callback=on_iter, epoch_callback=on_epoch)
```

The epoch callback printed, at the time of writing:

```text
epoch 1: test energy 224.4, train energy 120.3
epoch 2: test energy 192.1, train energy 299.6
epoch 3: test energy 265.8, train energy 355.9
epoch 4: test energy 317.8, train energy 366.8
```

The training energy rose after the first epoch instead of falling. Step 6 reads why.

Under an `InferenceSchedule`, `structure.config["inference"]` is the schedule and not an `EPCInference`, so pass the ePC segment explicitly: `RegimeProbe(structure, clamps_a, every=10, inference=epc, key=probe_key)`. Without it the regime columns stay blank.

### Step 6: Read the probe

`summary()` dates the first reversal and the first crossing and lists λ_max per epoch; the flag string below prints one character per probe. `make_inference_history` runs one settle and returns the state after every step, and `graph_energy` sums a state's energy, so the two together show what the flag means for the inner loop.

```python
from fabricpc.core.energy import graph_energy
from fabricpc.training import read_regime_csv
from fabricpc.utils.dashboarding import make_inference_history

print(probe.summary())
print("".join("!" if r["unstable"] else ("r" if r["output_gradient_reverses"] else ".") for r in probe.probe_rows()))
print("lambda_max per probe:", " ".join(f"{r['lambda_max']:.3g}" for r in probe.probe_rows()[:8]))
print("train energy per probe:", " ".join(f"{r['train_energy']:.3g}" for r in probe.probe_rows()[:8]))
metadata, rows = read_regime_csv("epc_regime_track.csv")
print(structure.config["inference"].regime(spectrum_at(structure, result.params, clamps_a)))


def settle_energies(structure, params, clamps):
    """Per-sample energy after each inference step of one settle on ``clamps``."""
    history = make_inference_history(structure)
    state0 = initialize_graph_state(structure, BATCH, jax.random.PRNGKey(1), clamps=clamps, params=params)
    _, states = history(params, state0, clamps)
    n_steps = states.nodes["y"].energy.shape[0]
    return [float(graph_energy(jax.tree_util.tree_map(lambda a: a[i], states), structure)) / BATCH
            for i in range(n_steps)]


print("settle at the end:            ", " ".join(f"{e:.4g}" for e in settle_energies(structure, result.params, clamps_a)))
print("settle at the last clear probe:", " ".join(f"{e:.4g}" for e in settle_energies(structure, last_clear["params"], clamps_a)))
```

Output at the time of writing:

```text
output-gradient reversal (odd T, eta*(lambda_max - 1) > 1 at T = 1): never flagged
stability bound eta*lambda_max > 2: first crossed at update 50 (epoch 1)
lambda_max per epoch (max over probes; ratio to previous epoch):
  epoch   1:        33.77
  epoch   2:        36.67  x1.09
  epoch   3:        36.23  x0.99
  epoch   4:        36.42  x1.01
....!!!!!!!!!!!!!!!!!!!!
lambda_max per probe: 11 14.3 19.5 24.7 32.1 33.8 32.5 35.5
train energy per probe: 124 76.6 44 23.8 86.4 213 156 371
unstable: eta*lambda_max = 2.11 > 2
settle at the end:             376.1 385.6 396.8 409.4 423.5
settle at the last clear probe: 360.8 81.83 42.25 30.96 27.21
```

How to read it. λ_max was 9.4 at init and 32 at the fifth probe, update 50, so the rate chosen at 0.3 of the init bound stood above the bound from that update on; the flag string marks the crossing at the fifth probe and every probe after it. The training energy had fallen from 124 to 24 over the first four probes and jumped to 86 and then 213 at the next two, and the test energy rose from the second epoch on. The settle at the final parameters raises the energy at every one of its four steps, from 376 to 424: the inner loop no longer minimizes the energy, which is what the flag predicts. The settle at the last clear probe, forty updates in, fell from 361 to 27 over the same four steps. In a row with `unstable` True the `f_weighted` and `f_max` columns are the formula evaluated past its range and mean nothing; `str(regime)` prints the unstable form ahead of them. `probe.first_chance(chance)` dates the epoch a classifier's accuracy reached chance, and `scripts/epc_analysis.py --plot_track epc_regime_track.csv` renders the CSV.

**On nonlinear models.** The bound is the quadratic model's at ε = 0, and on a linear-Gaussian graph that model is the whole energy, so the flag and the divergence coincide. On a graph with bounded curvature, such as sigmoid layers feeding a softmax cross-entropy output, the curvature falls away from ε = 0 and a settle can still converge past the bound: on the quickstart's MNIST perceptron the flag fired within two epochs of training while the twenty-step settle at the flagged parameters still decreased the energy at every step. The flag then marks where the guarantee ends, not a certain failure. Check the settle history as above before acting, and remember that the deep demo, also nonlinear, did fall to chance within an epoch of its crossing (report Section 5.9, Control runs).

### Step 7: Act on the flags

The signals in the order they arrive, and what each means:

| Signal | Meaning | Where measured |
|---|---|---|
| f̄ leaving the band you chose | λ_max is growing at fixed η·T, so the run drifts toward the equilibrium; on the demo the crossing of 0.1 preceded the collapse by epochs | report Section 5.9 |
| `output_gradient_reverses` | odd T only; the output layer's weight gradient points away from the target along the top mode; on the demo it fired one probe interval before the stability crossing | report Section 6.3 |
| `unstable` | η·λ_max > 2 at every T; the top mode grows every step from that update | `Regime.unstable` |
| `growth_min` > 1.1 | the energy is indefinite at this state and the negative-curvature modes that carry gradient weight grow over T steps | `Regime.growth_min` |

#### Remedies

Each remedy has a trade-off; the user decides which fits their model.

- **Lower η.** The one remedy the report measured: at fixed T a lower η removed the runaway within the tracked epochs on the demo (Section 5.9). Lowering η lowers f̄ at the same T, so the run may move into the backprop-like band; re-tabulate the grid at the current spectrum, re-read the label, and report the regime the run trained in, not the one you chose.
- **Compose ePC then sPC.** Run the ePC segment at a low η·T as a warm start and let a state-based segment finish the settle on the exact graph energy at its own per-node rate ([Composing ePC and sPC](#composing-epc-and-spc)).
- **Raise T** to hold f̄ after lowering η, or lower T to hold η·T·λ_max in the band you want as λ_max grows. T moves f̄; it does not move the `unstable` flag, which depends on η alone.
- **Restart from saved state.** Continue from the parameters and optimizer state of the last probe the flags cleared with `train(..., opt_state=..., start_epoch=...)`; the optimizer moments carry over ([TrainResult and resume](08_training_and_evaluation.md#trainresult-and-resume)). The library has no save or load helper for these; the callback in Step 5 keeps them in memory, and Orbax writes them to disk (the troubleshooting FAQ).

The example applies the first, the third, and the last. The spectrum at the last clear probe gives the new bound; the grid at that spectrum shows that 0.1 of it at T = 4 lands below the band the run had, and that T = 20 restores it. The same spectrum also shows what an odd T would have flagged at that point.

```python
snapshot = spectrum_at(structure, last_clear["params"], clamps_a)
lam_snapshot = float(snapshot.lambda_max)
print(f"last clear probe: update {last_clear['step']}, lambda_max {lam_snapshot:.4g}, new bound {2 / lam_snapshot:.4g}")
print("had T been 5:", EPCInference(eta_infer=eta_chosen, infer_steps=5).regime(snapshot))
eta_lower = round(0.1 * 2.0 / lam_snapshot, 4)
for T in (4, 20):
    print(f"eta {eta_lower}, T = {T}:", EPCInference(eta_infer=eta_lower, infer_steps=T).regime(snapshot))

structure = build(EPCInference(eta_infer=eta_lower, infer_steps=20))
probe = RegimeProbe(structure, clamps_a, every=10, key=probe_key)
result = train(last_clear["params"], structure, train_batches, optax.adam(3e-3), {"num_epochs": 4}, train_key,
               opt_state=last_clear["opt_state"], start_epoch=last_clear["epoch"], verbose=False,
               iter_callback=probe.on_iter, epoch_callback=on_epoch)
print(probe.summary())
print(structure.config["inference"].regime(spectrum_at(structure, result.params, clamps_a)))
print("settle at the end:", " ".join(f"{e:.4g}" for e in settle_energies(structure, result.params, clamps_a)[::4]))
```

```text
last clear probe: update 40, lambda_max 24.68, new bound 0.08103
had T been 5: eta*T*lambda_max = 7.86 (gradient-weighted relaxed fraction 1.03, fastest mode 1.06): near PC equilibrium; output-layer gradient reverses on the top mode (eta*(lambda_max - 1) = 1.51)
eta 0.0081, T = 4: eta*T*lambda_max = 0.8 (gradient-weighted relaxed fraction 0.50, fastest mode 0.59): partially relaxed
eta 0.0081, T = 20: eta*T*lambda_max = 4 (gradient-weighted relaxed fraction 0.95, fastest mode 0.99): near PC equilibrium
epoch 1: test energy 74.62, train energy 11.78
epoch 2: test energy 10.79, train energy 1.403
epoch 3: test energy 1.053, train energy 0.1431
epoch 4: test energy 0.07639, train energy 0.01187
output-gradient reversal (odd T, eta*(lambda_max - 1) > 1 at T = 1): never flagged
stability bound eta*lambda_max > 2: never crossed
lambda_max per epoch (max over probes; ratio to previous epoch):
  epoch   1:        45.38
  epoch   2:        54.02  x1.19
  epoch   3:        57.12  x1.06
  epoch   4:        58.02  x1.02
eta*T*lambda_max = 9.4 (gradient-weighted relaxed fraction 1.00, fastest mode 1.00): near PC equilibrium
settle at the end: 0.08051 0.007711 0.003453 0.002698 0.002459 0.002359
```

How to read it. At the last clear probe λ_max was 24.7 and the new bound 0.081, so the chosen rate of 0.0637 already stood at four fifths of it; one probe interval later it was past. Had T been 5, the reversal flag would have been on at that probe, with η·(λ_max − 1) = 1.51: the output layer's gradient along the top mode pointed away from the target while the run was still stable. In a run of this chain at T = 5 (not shown) that flag stayed on from the first epoch while the hidden layers compensated and the fit still progressed, more slowly; the flag is a warning about one layer's gradient along one mode, and on the deep demo it preceded the collapse by one probe interval. At 0.1 of the new bound, T = 4 reads partially relaxed with f̄ = 0.50, below the band the run had; T = 20 restores it. The restart from the saved state at (0.0081, 20) fit the teacher: the test energy fell from 75 to 0.08 over four epochs, no flag fired, and λ_max grew from 25 to 58 as the weights grew into the fit, its ratio per epoch falling toward 1 as the fit completed. At the end η·λ_max is 0.47 and the settle converges. Growth stopped here because the task has a finite solution; on the demo it did not stop (report Section 5.9).

#### Limits

The probe detects; it does not control. A rate that follows λ_max during training, lowering η as it grows or stopping with a diagnosis as η·λ_max approaches 2, is follow-up work (report Section 6.4, Follow-up). Until it exists, the user reads the flags and applies a remedy by hand, and the schedule and hyperparameters that work for a model are the user's to determine from these measurements.

### Step 8: Report the run

State: the solver or schedule and its (η, T); λ_max at init and at the end, with the probe batch size and the key; `str(regime)` at init and at the end; whether any flag fired, and at which update and epoch; and the band by name. A run whose label reads backprop-like is reported as backprop-like, whatever the solver was called.

## Composing ePC and sPC

`InferenceSchedule(EPCInference(eta_epc, T_epc), InferenceSGD(eta_spc, T_spc))` runs the ePC segment first and hands its derived latents to the state-based segment, which continues on the exact graph energy from that state. The two segments have two rate scales: η_epc is the global rate bounded by λ_max of your graph and chosen from the table above; η_spc is the per-node state-based rate, tuned by the state-based guidance in the [Inference Algorithms API](12_api_inference.md#tuning-guidance). The warm start carries the loss signal to every layer in a few global steps, so the state-based segment starts near the equilibrium instead of at the feedforward point. The composed schedule descends the energy across the segment boundary (`tests/test_inference_schedule.py::TestExecution::test_schedule_descends_energy`). On a cyclic graph the ePC segment minimizes the unrolled energy and the state-based segment the exact one, so the composition ends on the exact energy.

How many state-based steps the warm start saves is not measured in the report. Measure it on your graph with the settle history from Step 6, at equal step budgets. The block below does so on a six-hidden-layer chain at init, where the state-based solver needs a step per hop before the output clamp reaches the first layer.

```python
from fabricpc.core.inference import InferenceSGD, InferenceSchedule

deep = build(EPCInference(eta_infer=0.01, infer_steps=4), hidden=6)
deep_params = initialize_params(deep, graph_key)
deep_clamps = build_clamps(test_batches[0], deep, clamp_target=True)
deep_spectrum = spectrum_at(deep, deep_params, deep_clamps)
lam_deep = float(deep_spectrum.lambda_max)
eta_epc, T_epc, eta_spc, T_spc = round(0.3 * 2.0 / lam_deep, 4), 5, 0.1, 20
epc = EPCInference(eta_infer=eta_epc, infer_steps=T_epc)
print(f"six hidden layers: lambda_max {lam_deep:.4g}; ePC segment: {epc.regime(deep_spectrum)}")


def energy_after(inference):
    return settle_energies(build(inference, hidden=6), deep_params, deep_clamps)[-1]


print(f"sPC alone, {T_epc + T_spc} steps:     {energy_after(InferenceSGD(eta_infer=eta_spc, infer_steps=T_epc + T_spc)):.4g}")
print(f"sPC alone, 100 steps:    {energy_after(InferenceSGD(eta_infer=eta_spc, infer_steps=100)):.4g}")
print(f"ePC {T_epc} then sPC {T_spc} steps:  {energy_after(InferenceSchedule(epc, InferenceSGD(eta_infer=eta_spc, infer_steps=T_spc))):.4g}")
print(f"ePC alone, {T_epc + T_spc} steps:     {energy_after(EPCInference(eta_infer=eta_epc, infer_steps=T_epc + T_spc)):.4g}")
print(f"equilibrium (sPC, 400): {energy_after(InferenceSGD(eta_infer=eta_spc, infer_steps=400)):.4g}")

schedule = build(InferenceSchedule(epc, InferenceSGD(eta_infer=eta_spc, infer_steps=T_spc)), hidden=6)
print(RegimeProbe(schedule, deep_clamps, every=10, key=probe_key).inference)
print(RegimeProbe(schedule, deep_clamps, every=10, inference=epc, key=probe_key).inference is epc)
```

Output at the time of writing, per-sample energy on the probe batch:

```text
six hidden layers: lambda_max 25.59; ePC segment: eta*T*lambda_max = 2.99 (gradient-weighted relaxed fraction 0.88, fastest mode 0.99): partially relaxed
sPC alone, 25 steps:     140.2
sPC alone, 100 steps:    92.73
ePC 5 then sPC 20 steps:  91.01
ePC alone, 25 steps:     79.14
equilibrium (sPC, 400): 78.67
None
True
```

How to read it. On the six-layer chain the state-based solver alone needs a step per hop before the clamp reaches the first layer: 25 steps reach 140 and 100 steps 92.7, against an equilibrium of 78.7. Five ePC steps at 0.3 of the bound followed by twenty state-based steps reach 91.0, which the state-based solver alone needs about a hundred steps for, so the warm start cut the state-based budget by about four times at that energy. Twenty-five ePC steps alone reach 79.1, close to the equilibrium: on a linear chain ePC alone at a safe η is the fastest route to the equilibrium, and the composition's case is a cyclic graph, where it ends on the exact energy, or an ePC segment kept far below the bound while the state-based segment finishes. Which holds for your graph is the measurement above. The probe needs the ePC segment as `inference=epc`, as the last two lines show, and the regime label then describes that segment's relaxation, not the schedule's final state; the state-based segment is judged by its own settling history ([Experiment Tracking](09_experiment_tracking.md)).

## Reading the regime line

`str(regime)` has three forms, in precedence order. The examples are from the chain above.

1. **Unstable**, when η·λ_max > 2:

   ```text
   unstable: eta*lambda_max = 2.11 > 2
   ```

2. **Indefinite**, when `growth_min` > 1.1 (the chain's energy is positive definite, so this form did not appear above; the format is):

   ```text
   indefinite: negative curvature carrying {percent}% of the gradient grows {growth_min}x over {T} steps (lambda_min = {lambda_min})
   ```

3. **The band**, with f̄ and f_max, and the reversal note appended when `output_gradient_reverses` is True:

   ```text
   eta*T*lambda_max = 0.02 (gradient-weighted relaxed fraction 0.01, fastest mode 0.02): backprop-like
   eta*T*lambda_max = 2.4 (gradient-weighted relaxed fraction 0.79, fastest mode 0.97): partially relaxed
   eta*T*lambda_max = 7.86 (gradient-weighted relaxed fraction 1.03, fastest mode 1.06): near PC equilibrium; output-layer gradient reverses on the top mode (eta*(lambda_max - 1) = 1.51)
   ```

The band has four values: `"backprop-like"`, `"partially relaxed"`, `"near PC equilibrium"`, and `"no positive curvature"` when no positive mode carries gradient weight. A relaxed fraction above 1, as in the third line, means the modes overshot their equilibrium within the T steps, which happens for 1 < ηλ < 2; the band still reads near equilibrium, and at odd T the reversal note is the warning that the output layer is being trained the wrong way along the top mode.

## Optimizer interaction

In the backprop-like band the hidden-layer weight gradients are backprop's scaled by η and the output layer's are unscaled. Adam normalizes the scale away while η·|g| ≫ Adam's ε (1e-8), so a one-step ePC run with Adam trains as backprop with Adam; below that floor the ε term damps the hidden layers. Without Adam the hidden layers learn η times slower than the output layer. At the equilibrium the output error is re-weighted by the network's transfer function, a matrix rescaling Adam cannot undo (report Sections 5.3 and 5.10). This guide's recommendation: Adam or AdamW under `EPCInference`, unless the rate disparity between the layers is itself the object of study.

## Failure signatures

| What you see | What it is | Where to look |
|---|---|---|
| Accuracy improves for epochs, then falls to chance within a few | λ_max grew until η·λ_max crossed 2 | Step 7 |
| The ePC arm's accuracy equals the backprop arm's | the run is in the backprop-like band | Step 4, Step 8 |
| Chance from the first epoch at T = 1 or another odd T | η·(λ_max − 1) > 1: the output gradient is reversed from the first update | Step 3 (`r` in the grid) |
| Hidden layers barely learn under SGD; the output layer does | the η scaling of the hidden-layer gradients, without Adam | Optimizer interaction |
| The label changed to backprop-like after lowering η | f̄ falls with η at fixed T | Step 7, Remedies |
| The probe CSV has blank regime columns | the configured inference is a schedule, or the trainer is backprop | Step 5 |

The same table, with causes and actions spelled out, is in [Troubleshooting](16_troubleshooting.md#epc-epcinference).

## Where the numbers are

The measurements this guide points to are in `docs/reports/epc_regime_and_stability_report.md`: the spectrum at init and the 2-epoch sweep and the 100-epoch runs on the demo (Section 5.8), the control runs that dated the collapse against the spectrum and the weight norms (Section 5.9), the stability bound across init scales and depths (Section 5.7), the equilibrium damping (Section 5.10), the paper's ResNet-18 against the demo (Section 5.11), Limits (Section 6.3), and Follow-up (Section 6.4). `examples/resnet18_cifar10_demo.py --track_regime N` reproduces the probe on the demo, and `scripts/epc_analysis.py --plot_track` renders any probe CSV.
