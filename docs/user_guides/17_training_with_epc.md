# Training with ePC

`EPCInference` relaxes the prediction errors instead of the latent states: one reverse pass per step carries the loss signal to every layer, so a few steps replace the hundreds a state-based solver needs on a deep graph (update rule and hooks: [Inference Algorithms API](12_api_inference.md#epcinference)). Its rate η and step count T are yours to choose, and three facts decide the choice:

- **η is one global rate through the whole network.** Relaxation is stable only below 2/λ_max, where λ_max is the largest excited eigenvalue of the energy Hessian in error coordinates, a property of your graph at your init. Rates from the state-based guides, or from another graph, do not carry over.
- **λ_max grows during training**, so a rate that is safe at init has no lasting margin (report Section 6.2). The probe below re-measures it as you train.
- **One ePC step from zero error is backprop's activation gradient.** Whether a run is backprop-like, partially relaxed, or near the PC equilibrium is read from f̄, the gradient-weighted relaxed fraction, and the library defaults (1e-3, 5) were backprop-like at init on the ResNet-18 demo. Read the label before reporting a run as PC.

The workflow below runs on the quickstart's MNIST perceptron; your graph gives other numbers, and the procedure derives them from your graph. The deep-network measurements are in the report, `docs/reports/epc_regime_and_stability_report.md`, cited by section.

## Symbols

| Symbol | Meaning | Where |
|---|---|---|
| η, T | `eta_infer`, `infer_steps` | `EPCInference` |
| λ_max, λ_min | largest and smallest eigenvalues of the energy Hessian in error coordinates, over the modes the starting gradient excites | `epsilon_spectrum` |
| f(λ) | 1 − (1 − ηλ)^T, how far a mode of eigenvalue λ has relaxed toward its equilibrium after T steps | formula |
| f̄ | f averaged over the positive modes, each weighted by its share of the starting gradient | `Regime.f_weighted` |
| f_max | f(λ_max) | `Regime.f_max` |
| `negative_weight` | share of the starting gradient on negative-curvature modes | spectrum, `Regime` |
| `growth_min` | growth of the most negative mode over T steps | `Regime` |

The band is read on f̄: below 0.1 backprop-like, above 0.9 near PC equilibrium, between them partially relaxed.

## Workflow

### Step 1: Build the graph with an explicit solver

The quickstart's graph, with the solver as the only change. The pair is provisional until Step 4.

```python
import jax
import optax
from fabricpc import setup_jax
from fabricpc.core import EPCInference
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.utils.data.dataloader import MnistLoader

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

pixels = IdentityNode(shape=(784,), name="pixels")
hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1", weight_init=XavierInitializer())
hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2", weight_init=XavierInitializer())
output = Linear(shape=(10,), activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class",
                weight_init=XavierInitializer())


def build(inference):
    return graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[Edge(source=pixels, target=hidden1.slot("in")), Edge(source=hidden1, target=hidden2.slot("in")),
               Edge(source=hidden2, target=output.slot("in"))],
        task_map=TaskMap(x=pixels, y=output),
        inference=inference,
    )


graph_key, train_key, probe_key = jax.random.split(jax.random.PRNGKey(0), 3)
structure = build(EPCInference(eta_infer=0.01, infer_steps=5))  # provisional
params = initialize_params(structure, graph_key)
batch_size = 200
train_loader = MnistLoader("train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42)
test_loader = MnistLoader("test", batch_size=batch_size, tensor_format="flat", shuffle=False)
```

### Step 2: Measure the spectrum at init

The spectrum is measured at the feedforward state of one clamped batch.

```python
from fabricpc.core import epsilon_spectrum
from fabricpc.graph_initialization import initialize_graph_state
from fabricpc.training import build_clamps
from fabricpc.training.trainer import convert_batch

probe_clamps = build_clamps(convert_batch(next(iter(test_loader))), structure, clamp_target=True)


def spectrum_at(structure, params):
    state = initialize_graph_state(structure, batch_size, jax.random.PRNGKey(1), clamps=probe_clamps, params=params)
    return epsilon_spectrum(params, state, probe_clamps, structure)


spectrum = spectrum_at(structure, params)
lam_max = float(spectrum.lambda_max)
```

At the time of writing: λ_max = 1.40, so the bound 2/λ_max = 1.43; λ_min = 0.92 and `negative_weight` = 0, so the energy is positive definite at this state. This is one batch at one state: measure a second batch and take the spread as the bound's uncertainty (under the default `FeedforwardStateInit` the key does not change the state). λ_min from 30 Lanczos steps is a sign and an order of magnitude; a negative value with nonzero `negative_weight` means the energy is indefinite here. On another graph the bound may be a thousand times smaller (report Section 5.7).

### Step 3: Tabulate the regime for a grid

`EPCInference(eta, T).regime(spectrum)` is a formula on the measured spectrum, so the band of any pair costs nothing. Tabulate η as a fraction of the bound, so the table transfers across graphs.

```python
for fraction in [0.01, 0.03, 0.1, 0.3]:
    for T in [1, 5, 10, 20, 50]:
        r = EPCInference(eta_infer=fraction * 2 / lam_max, infer_steps=T).regime(spectrum)
        print(fraction, T, r.band, round(r.f_weighted, 2))
```

The output, arranged as a table (band and f̄):

| fraction of bound | T = 1 | T = 5 | T = 10 | T = 20 | T = 50 |
|---|---|---|---|---|---|
| 0.01 | backprop-like 0.02 | backprop-like 0.08 | partially relaxed 0.16 | partially relaxed 0.29 | partially relaxed 0.58 |
| 0.03 | backprop-like 0.05 | partially relaxed 0.23 | partially relaxed 0.41 | partially relaxed 0.65 | near PC equilibrium 0.93 |
| 0.10 | partially relaxed 0.17 | partially relaxed 0.61 | partially relaxed 0.85 | near PC equilibrium 0.98 | near PC equilibrium 1.00 |
| 0.30 | partially relaxed 0.52 | near PC equilibrium 0.97 | near PC equilibrium 1.00 | near PC equilibrium 1.00 | near PC equilibrium 1.00 |

Your table is your range: rows further down have less margin to the bound, columns further right cost more reverse passes per update. This guide gives no fixed numeric range for η or T.

### Step 4: Choose η and T

- **Backprop-like** (f̄ < 0.1): the weight gradients are backprop's, hidden layers scaled by η, which Adam normalizes away. Fastest per update; the result is a backprop result.
- **Near PC equilibrium** (f̄ > 0.9): the errors have settled. Needs η·T·λ ≳ 3 on the modes that carry the gradient. At the equilibrium the output error is re-weighted by the network's transfer function, and on the demo the equilibrium cells reached a lower accuracy plateau (report Section 5.10, a linear-chain argument).
- **Partially relaxed**: between.

Read both f̄ and f_max; the band is on f̄, and whether f_max should label the backprop-like edge is open (report Section 6.4). Start well below the bound: growth, not the margin at init, decides the run. The example takes a tenth of the bound with T = 20.

```python
eta, T = round(0.1 * 2 / lam_max, 4), 20
structure = build(EPCInference(eta_infer=eta, infer_steps=T))
params = initialize_params(structure, graph_key)
print(structure.config["inference"].regime(spectrum))
```

```text
eta*T*lambda_max = 4 (gradient-weighted relaxed fraction 0.98, fastest mode 0.99): near PC equilibrium
```

### Step 5: Attach the probe

`RegimeProbe` re-measures the spectrum every `every` weight updates on the probe batch and records the flags, the weight norms, and the accuracy you pass to `on_epoch`. `evaluate` returns a dict; pass its `"accuracy"`. Any `iter_callback` forces a device sync per batch; the probe itself is one initialization and 30 Hessian-vector products per probe. Under an `InferenceSchedule`, pass the ePC segment as `inference=`, or the regime columns stay blank.

```python
from fabricpc.training import RegimeProbe, evaluate, train

probe = RegimeProbe(structure, probe_clamps, every=50, key=probe_key, csv_path="epc_regime_track.csv")
optimizer = optax.adamw(1e-3, weight_decay=0.1)


def on_epoch(ctx):
    accuracy = evaluate(ctx.params, ctx.structure, test_loader, ctx.config, ctx.epoch_key)["accuracy"]
    probe.on_epoch(ctx, accuracy)


result = train(params, structure, train_loader, optimizer, {"num_epochs": 2}, train_key,
               verbose=False, iter_callback=probe.on_iter, epoch_callback=on_epoch)
```

### Step 6: Read the probe

```python
print(probe.summary(chance=0.1))
print(structure.config["inference"].regime(spectrum_at(structure, result.params)))
```

```text
output-gradient reversal (odd T, eta*(lambda_max - 1) > 1 at T = 1): never flagged
stability bound eta*lambda_max > 2: never crossed
test accuracy within 0.05 of chance (0.1): never
lambda_max per epoch (max over probes; ratio to previous epoch):
  epoch   1:        4.576
  epoch   2:        6.537  x1.43
eta*T*lambda_max = 18.6 (gradient-weighted relaxed fraction 0.99, fastest mode 1.00): near PC equilibrium
```

λ_max went from 1.4 at init to 6.5 after two epochs while the test accuracy rose to 94.6% (the CSV's `test_accuracy` column, read back with `read_regime_csv`). No flag fired, but the margin to the bound shrank from ten times to about two, and at this growth the bound would be crossed within a few more epochs. The probe, not the margin at init, says when to act.

`str(regime)` has three forms, in precedence order: `unstable: eta*lambda_max = ... > 2`; `indefinite: negative curvature carrying ...% of the gradient grows ...x over T steps` when `growth_min` > 1.1; otherwise the band line above, with `; output-layer gradient reverses on the top mode (...)` appended when that flag is set (odd T only). A fourth band, `no positive curvature`, appears when no positive mode carries gradient weight. In a CSV row with `unstable` True, `f_weighted` and `f_max` are the formula past its range and mean nothing.

### Step 7: Act on the flags

| Signal | Meaning |
|---|---|
| f̄ leaving the band you chose | λ_max is growing at fixed η·T; the run drifts toward the equilibrium. On the demo it led the collapse by epochs (report Section 5.9) |
| `output_gradient_reverses` | odd T only: the output layer's gradient points away from the target along the top mode. On the demo it led the crossing by one probe interval (Section 6.3) |
| `unstable` | η·λ_max > 2 at every T: the top mode grows every step from that update |
| `growth_min` > 1.1 | the energy is indefinite here and the negative modes that carry gradient grow over T steps |

Remedies, each with its trade-off:

- **Lower η.** The remedy the report measured (Section 5.9). It lowers f̄ at the same T, so the run may leave the band you chose; re-read the label and report the regime the run trained in.
- **Raise T** to hold f̄ after lowering η. T moves f̄, not the `unstable` flag, which depends on η alone.
- **Compose ePC then sPC**: keep the ePC segment far below the bound and let a state-based segment finish the settle (next section).
- **Restart from saved state**: `train(..., opt_state=result.opt_state, start_epoch=...)` continues with the optimizer moments intact ([TrainResult and resume](08_training_and_evaluation.md#trainresult-and-resume)). Save `result.params` and `result.opt_state` at the last good epoch yourself; the library has no helper.

The mechanics, continuing the run above at a lower η for one epoch:

```python
eta = round(0.03 * 2 / float(spectrum_at(structure, result.params).lambda_max), 4)
structure = build(EPCInference(eta_infer=eta, infer_steps=T))
result = train(result.params, structure, train_loader, optimizer, {"num_epochs": 1}, train_key,
               opt_state=result.opt_state, start_epoch=2, verbose=False, epoch_callback=on_epoch)
print(structure.config["inference"].regime(spectrum_at(structure, result.params)))
```

```text
eta*T*lambda_max = 1.63 (gradient-weighted relaxed fraction 0.50, fastest mode 0.82): partially relaxed
```

The lower rate moved the run from near the equilibrium to partially relaxed at the same T. To hold the band, raise T; to report the run, name the band it trained in.

The flags are the local quadratic model's predictions at ε = 0. On a linear-Gaussian graph they are exact. On this perceptron, in a run at a higher rate (not shown), `unstable` fired while the settle still converged, because the curvature of sigmoid layers and a softmax output falls away from ε = 0; on the deep demo the run fell to chance within an epoch of its crossing. When a flag fires on a nonlinear graph, look at the settle's energy per step (`make_inference_history`, [Experiment Tracking](09_experiment_tracking.md)) before deciding. The library detects; it does not control. A rate that follows λ_max is follow-up work (report Section 6.4).

### Step 8: Report the run

State the solver or schedule and its (η, T); λ_max at init and at the end with the probe batch size; `str(regime)` at init and at the end; any flag with its update and epoch; and the band by name. A run whose label reads backprop-like is reported as backprop-like.

## Composing ePC and sPC

`InferenceSchedule(EPCInference(...), InferenceSGD(...))` runs the ePC segment first and hands its latents to the state-based segment, which continues on the exact graph energy at its own per-node rate. The warm start carries the loss signal to every layer in a few global steps, so the ePC segment can stay far below the bound while the state-based segment finishes the settle; on a cyclic graph the composition ends on the exact energy. How many state-based steps it saves is not measured in the report. Measure it on your graph: settle once with each solver at an equal step budget and compare the energies.

```python
from fabricpc.core.energy import graph_energy
from fabricpc.core.inference import InferenceSGD, InferenceSchedule, run_inference


def settled_energy(inference):
    s = build(inference)
    state = initialize_graph_state(s, batch_size, jax.random.PRNGKey(1), clamps=probe_clamps, params=params)
    return float(graph_energy(run_inference(params, state, probe_clamps, s), s)) / batch_size


epc = EPCInference(eta_infer=round(0.03 * 2 / lam_max, 4), infer_steps=5)
print(settled_energy(InferenceSGD(eta_infer=0.05, infer_steps=25)),
      settled_energy(InferenceSchedule(epc, InferenceSGD(eta_infer=0.05, infer_steps=20))),
      settled_energy(EPCInference(eta_infer=epc.config["eta_infer"], infer_steps=25)))
```

At init, per sample on the probe batch: sPC alone for 25 steps reaches 1.844, five ePC steps then twenty sPC steps 1.841, and ePC alone for 25 steps 1.835. On this three-layer graph the three end within half a percent of each other; the saving the warm start is designed for is on deep graphs, where the state-based solver needs a step per hop before the output clamp reaches the first layer. The probe under a schedule needs `inference=epc`; its label then describes the ePC segment's relaxation, not the schedule's final state.

## Optimizer interaction

In the backprop-like band the hidden-layer weight gradients are backprop's scaled by η and the output layer's are unscaled. Adam normalizes the scale away while η·|g| ≫ Adam's ε of 1e-8; without Adam the hidden layers learn η times slower than the output layer (report Section 5.3). At the equilibrium the output error is re-weighted by the network's transfer function, which Adam cannot undo. This guide's recommendation: Adam or AdamW under `EPCInference`, unless the disparity is the object of study.

## Where to look next

Symptoms with causes and actions are in [Troubleshooting](16_troubleshooting.md#epc-epcinference). The report holds the measurements this guide points to: the ResNet-18 spectrum, sweep, and 100-epoch runs (Section 5.8), the control runs that dated the collapse against the spectrum (5.9), the stability bound across init scales and depths (5.7), the equilibrium damping (5.10), Limits (6.3), and Follow-up (6.4). `examples/resnet18_cifar10_demo.py --track_regime N` runs the probe on the demo, and `scripts/epc_analysis.py --plot_track` renders any probe CSV.
