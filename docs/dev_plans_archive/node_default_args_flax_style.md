# Standardize node default arguments on Flax-style signature defaults

**Status: implemented.** Landed across four commits: `88b0b7d` (Conv/Pool merged
with signature defaults already applied, pre-merge), `00fde43` (freeze +
ruff + `transformer.py`), `a7d5ac0` (`transformer_v2.py` migration), `c50ab0d`
(older nodes, docs, changelog). This document was updated after implementation
to record the design choices that diverged from the original plan; each is
marked **Design choice (changed from plan)**.

## Context

At planning time the codebase was split between two default-argument
conventions. The pre-merge working copy of the Conv/Pool PR filled its
object-valued defaults — `activation`, `energy`, `weight_init`, `latent_init`,
`bias_init` — in the constructor **body** (`param=None` in the signature, then
`if param is None: param = Default()`), with the comment:

> `# Fill defaults in the body — never in the signature — to avoid the mutable-default-argument pitfall.`

`transformer_v2.py` used the equivalent `param or Default()` body form. The
older nodes (`linear.py`, `identity.py`, `skip_connection.py`,
`linear_residual.py`, `linear_explicit_grad.py`, `storkey_hopfield.py`,
`transformer.py`) placed the same defaults **in the signature**
(e.g. `linear.py` `activation: Optional[ActivationBase] = IdentityActivation()`),
and `base.py`'s own docstrings taught the signature-default convention.

**Why the body comment overstates the danger.** The classic
mutable-default-argument bug is that a default expression is evaluated once at
import and the resulting object is shared by every defaulted call, so *mutating*
that shared object leaks state across callers. The default objects here
(`ReLUActivation`, `GaussianEnergy`, `KaimingInitializer`, `NormalInitializer`,
`ZerosInitializer`) carry no mutable state: each stores only
`self.config = types.MappingProxyType(config)` — a read-only mapping — and does
all computation in `@staticmethod`s that take `config` as a parameter. The
shared singleton was therefore safe *by incident* (nothing mutated it, and
`MappingProxyType` turns an attempted in-place edit into a `TypeError`), not
*by construction*.

This is exactly how **Flax** treats the same problem: its module defaults are
stateless functions/closures (`activation: Callable = nn.relu`,
`kernel_init = lecun_normal()`, `bias_init = zeros_init()`) placed directly in
the dataclass field / `__init__` signature, safe precisely because the defaulted
value holds no mutable state. FabricPC's config-objects are already effectively
stateless value objects — they behave like Flax's function defaults.

**Decision (chosen):** standardize the node package on Flax-style **signature
defaults**, and convert the objects' "safe by incident" property into "safe by
construction" by making the three base classes immutable. Add a linter
(`ruff` B006) to forbid the genuinely dangerous mutable-*container* defaults
while permitting the stateless-object defaults.

## Approach — as implemented

### 1. Freeze the three config base classes (safe by construction)

**Design choice (changed from plan): one shared mixin, not three copies.** The
plan showed the freeze (`__init__`/`__setattr__`/`__delattr__`) pasted into
each of `ActivationBase`, `EnergyFunctional`, and `InitializerBase`. The
implementation factors it into a single `FrozenConfig` mixin in
`fabricpc/core/_frozen.py:13`, which all three bases inherit
(`class ActivationBase(FrozenConfig, ABC)` — `activations.py:51`,
`energy.py:50`, `initializers.py:48`). One implementation cannot drift across
copies; `tests/test_immutable_config.py::test_freeze_is_single_source` asserts
all three families resolve to `FrozenConfig.__setattr__` / `__delattr__`.

```python
class FrozenConfig:
    def __init__(self, **config):
        object.__setattr__(self, "config", types.MappingProxyType(config))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")
```

Notes (confirmed in implementation):
- No concrete subclass writes any instance attribute other than `config`, so no
  subclass `__init__` needed adjustment.
- `__setattr__` is the right freeze mechanism because `copy.copy` /
  `copy.deepcopy` reconstruct via `__dict__` update / `__setstate__`, which
  bypass `__setattr__`. The node finalize path `_with_graph_info` does
  `copy.copy(self)` on the **node** (sharing the activation/energy/init objects
  by reference), so freezing those objects does not interfere with
  copy-on-finalize (`TestCopyOnFinalize` covers this).
- The `FrozenConfig` docstring states the freeze is **shallow**: a mutable value
  stored under a `config` key is not itself frozen, so subclasses must construct
  only with immutable scalar config values.
- Each base class docstring states instances are frozen and safe to share as a
  signature default.

### 2. Migrate the body-default nodes to signature defaults

Stateless-object defaults moved into the signature everywhere. The only
defaults that stay body-filled are **computed defaults** that depend on another
argument (`stride` derived from `len(shape)-1` in `ConvNode`; `stride`
defaulting to `window_shape` in the pooling nodes).

> **Note — `slots` argument.** The `slots` constructor argument was removed from
> `ConvNode` and the pooling nodes before the Conv/Pool PR merged; they define
> their input slots via the static `get_slots()` method like the other nodes.
> No `slots=None` + body fill remained to migrate.

**Design choice (changed from plan): drop `Optional[...]`, don't keep it.** The
plan said to match `linear.py`'s `Optional[ActivationBase]` annotation form.
Instead, `c50ab0d` removed `Optional` from every constructor parameter that in
fact always receives an instance — the annotation now states the invariant
(`activation: ActivationBase = IdentityActivation()`). `Optional` remains only
where `None` is a real value: weight-free nodes' `initialize_params`
(`weight_init: Optional[InitializerBase] = None` in `identity.py`,
`skip_connection.py`) and `TransformerBlock.internal_activation` (see below).
Annotation form otherwise follows each file's existing style: the older nodes
use plain names, `convolutional.py`/`pooling.py` use string forward references
under `TYPE_CHECKING`, `transformer_v2.py` constructors are unannotated.

**Design choice (added beyond plan): the constructor is the single source of
truth for defaults.** Several static `initialize_params` methods carried a
hidden second default that *conflicted* with the constructor's:
`Linear`/`LinearResidual` re-defaulted a `None` `weight_init` to
`NormalInitializer(mean=0.0, std=0.05)` (constructor default:
`KaimingInitializer()`); `StorkeyHopfield` re-defaulted to `ZerosInitializer()`
(constructor default: `XavierInitializer()`). These `if weight_init is None:`
branches were deleted and `weight_init` made a required parameter of
`initialize_params` — the value always flows in from the constructor via
`node_info.weight_init`. `NodeInfo`'s field comments (`core/types.py`) now
state the invariant: `latent_init` is always an instance; `weight_init` is
`None` only for weight-free nodes.

**`convolutional.py` (`ConvNode.__init__`, lines 82–131):** signature defaults
`activation=ReLUActivation()`, `energy=GaussianEnergy()`,
`weight_init=KaimingInitializer()`, `bias_init=ZerosInitializer()`,
`latent_init=NormalInitializer()`; `stride=None` + body fill (computed from
spatial rank). The body comment forbidding signature defaults is gone.
- **Design choice (added beyond plan): fail-fast bias guard.**
  `initialize_params` raises `ValueError` when `use_bias=True` and
  `bias_init is None` (`convolutional.py:168`), rather than passing `None` into
  `initialize()`. Normal construction never hits it; it guards a hand-built
  config or an explicit `bias_init=None`.
- **Design choice (changed from plan): no Kaiming gain-resolution block.** The
  plan assumed a body block that rebuilt the Kaiming initializer with an
  activation-derived gain. The merged `ConvNode` has none: the user-supplied
  `weight_init` flows through verbatim to the conv kernel shape, and pairing an
  initializer config (e.g. `nonlinearity="leaky_relu"`) with the chosen
  activation is the caller's responsibility (module docstring, Design notes).

**`pooling.py` (`MaxPool` line 184, `AvgPool` line 265):**
`activation=IdentityActivation()`, `energy=GaussianEnergy()`,
`latent_init=NormalInitializer()` in the signature; `stride`/`window_shape`
stay body-filled (computed: stride defaults to the window for non-overlapping
pooling; `global_pool=True` sets both to `()`).

**`transformer_v2.py` (5 nodes: `EmbeddingNode`, `MhaResidualNode`,
`LnMlp1Node`, `Mlp2ResidualNode`, `VocabProjectionNode`):** the
`param or Default()` body form replaced with signature defaults (`a7d5ac0`),
e.g. `EmbeddingNode`'s `weight_init=NormalInitializer(std=0.02)`. No computed
or container defaults remained in these constructors.

**`transformer.py` (`TransformerBlock`) — design choice (added beyond plan).**
The v1 transformer's `internal_activation: Optional[ActivationBase] = None` +
`internal_activation or GeluActivation()` body form was also migrated: the
signature default is now `GeluActivation()`, and the parameter stays
`Optional` because `None` is meaningful — `forward()`'s identity path
(`activation_fn(x) = x`), previously unreachable because the `or` idiom forced
a non-`None` value, is now selected by an explicit `internal_activation=None`.
Covered by
`test_transformer_block_internal_activation_default_and_none`.

**`core/energy.py` module functions — design choice (added beyond plan).** The
convention also applies outside node constructors: `compute_energy` and
`compute_energy_gradient` migrated from `energy=None` + body fill to
`energy: EnergyFunctional = GaussianEnergy()` in the signature.

### 3. Add `ruff` B006 — and explicitly not B008

Added in `00fde43`, as planned, with scoping decisions made during
implementation:

- `pyproject.toml` `[tool.ruff.lint]` sets `extend-select = ["B006"]`
  (mutable-argument-default: forbids `list`/`dict`/`set` literal defaults).
  B008 (function-call-in-default-argument) is intentionally **not** enabled —
  it would flag every `= GaussianEnergy()` signature default, which is exactly
  the pattern being standardized on. B006 does not flag custom constructor
  calls, so all migrated defaults pass clean.
- **Design choice (added beyond plan): ruff is scoped to `fabricpc/` only.**
  `extend-exclude` lists `examples`, `tests`, and `scripts`: the demo and
  diagnostic code deliberately configures JAX
  (`set_jax_flags_before_importing_jax`) before `import jax`, which is E402 by
  design, and `tests/` run under pytest and are not shipped.
- **Design choice (added beyond plan): default ruleset enforced too.**
  `extend-select` runs B006 on top of ruff's default rules (pyflakes F +
  pycodestyle E4/E7/E9); bringing the package clean against those removed
  unused imports and dead lines across `fabricpc/` in the same commit.
- `.pre-commit-config.yaml` adds the `ruff-check` hook (lint only) at rev
  `v0.15.19` alongside Black, which remains the formatter. `ruff==0.15.19` is
  pinned in the dev dependencies so local `ruff check` and the hook agree.
- Optional follow-up (not taken): because the config-objects are provably
  immutable, they could later be added to
  `lint.flake8-bugbear.extend-immutable-calls` if the team ever wants B008 too.

### 4. `base.py` docstrings

Implemented slightly wider than the planned single sentence: the
user-extensibility example in the module docstring now shows signature defaults
and closes with the convention statement (`base.py:33`) — default objects are
immutable, so the single default instance is safe to share across every node
that does not override it. The `NodeBase` class docstring gained the matching
paragraph (`base.py:173`).

### 5. Document the convention in `docs/user_guides`

Implemented as planned:

- **`06_custom_nodes.md`** — Step 1 "Key points" (lines 82–84): object-valued
  defaults go in the signature (the objects are immutable, so the module-level
  singleton cannot leak state between nodes); body fill only for defaults
  computed from another argument; never default a parameter to a
  `dict`/`list`/`set` literal — `ruff` B006 rejects it in CI.
- **`11_api_activations_and_energy.md`** (lines 118, 243) and
  **`13_api_initializers.md`** (line 69): one sentence each in the
  custom-subclass sections — instances are frozen after
  `super().__init__(**config)`; all configuration must pass through `config`;
  setting an instance attribute in a subclass `__init__` raises
  `AttributeError`.

No guide showed the body-fill constructor pattern, so no existing example
needed rewriting.

### 6. Changelog (added beyond plan)

`CHANGELOG.md` gained an Unreleased entry: base classes frozen at
construction; `None` guards removed and defaults migrated from method bodies to
constructors; `Optional` dropped from always-required constructor arguments;
ruff added to pre-commit.

## Critical files — as touched

- `fabricpc/core/_frozen.py` — new `FrozenConfig` mixin (single freeze source).
- `fabricpc/core/activations.py`, `energy.py`, `initializers.py` — bases
  inherit `FrozenConfig`; docstring freeze notes; `compute_energy` /
  `compute_energy_gradient` signature defaults.
- `fabricpc/core/types.py` — `NodeInfo` field comments state the
  never-`None` invariants.
- `fabricpc/nodes/convolutional.py`, `pooling.py` — signature defaults (landed
  pre-merge in the Conv/Pool PR, `88b0b7d`); `ConvNode` bias fail-fast guard.
- `fabricpc/nodes/transformer_v2.py` — 5 nodes, `or`-form → signature defaults.
- `fabricpc/nodes/transformer.py` — `internal_activation` signature default;
  explicit `None` now reaches the identity path.
- `fabricpc/nodes/linear.py`, `identity.py`, `skip_connection.py`,
  `linear_residual.py`, `linear_explicit_grad.py`, `storkey_hopfield.py` —
  `Optional` dropped; hidden `initialize_params` re-defaults removed.
- `pyproject.toml`, `.pre-commit-config.yaml` — ruff B006 (not B008), scoped to
  `fabricpc/`, version pinned to the hook rev.
- `fabricpc/nodes/base.py` — module and class docstrings state the convention.
- `docs/user_guides/06_custom_nodes.md`, `11_api_activations_and_energy.md`,
  `13_api_initializers.md` — convention and freeze notes.
- `CHANGELOG.md` — Unreleased entry.
- `tests/test_immutable_config.py` — new test module (see Verification).

## Verification — as implemented

`tests/test_immutable_config.py` implements the planned checks and two added
ones:

1. **Immutability is enforced** (`TestImmutability`): for one activation, one
   energy, one initializer — `obj.config["k"] = 1` raises `TypeError`;
   `obj.new_attr = 1`, `obj.config = {}`, and `del obj.config` raise
   `AttributeError`; construction-time config still lands
   (`GaussianEnergy(precision=2.0).config["precision"] == 2.0`).
2. **Shared default is safe** (`TestSharedDefaultIsSafe`): two `ConvNode`s and
   two `MhaResidualNode`s built without overrides share the same default
   objects by identity, and those objects reject mutation.
3. **Copy-on-finalize still works** (`TestCopyOnFinalize`): `copy.copy` of a
   node shares the frozen objects by reference; a conv→pool→linear graph and a
   transformer_v2 block graph finalize via `graph()` and run inference; the
   finalized nodes' frozen objects still reject mutation.
4. **ConvNode bias parity** (`test_conv_bias_parity`): `use_bias=True` yields a
   bias parameter, `use_bias=False` yields none — the same parameter sets as
   before the migration.
5. **Single-source freeze** (`test_freeze_is_single_source`, added): all three
   families resolve to `FrozenConfig.__setattr__` / `__delattr__`.
6. **`TransformerBlock` default vs explicit `None`**
   (`test_transformer_block_internal_activation_default_and_none`, added): the
   default is one shared frozen `GeluActivation` singleton; explicit
   `internal_activation=None` selects the identity path.

Plus the planned non-test checks: full `pytest` suite; `ruff check fabricpc/`
clean (zero B006 findings) and `pre-commit run --all-files` confirming the hook
wiring; the custom-subclass snippets in guides 06/11/13 construct cleanly
against the frozen bases (each calls only `super().__init__(**config)`).

## Scope — as landed

All nodes end on a single convention: **signature defaults** for every
stateless-object parameter, with the constructor as the only place a default is
defined. Body fill remains only for computed defaults (`stride`,
`window_shape`). The final scope was wider than planned: beyond migrating the
body-pattern files (`convolutional.py`, `pooling.py`, `transformer_v2.py`),
the older signature-default nodes were also touched — `Optional` dropped from
always-required parameters, conflicting hidden re-defaults deleted from their
static `initialize_params`, and `transformer.py`'s `or`-idiom migrated. The
convention is stated in `docs/user_guides` so external node authors follow it
too.
