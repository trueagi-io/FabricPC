# Standardize node default arguments on Flax-style signature defaults

**Status: implemented, then hardened after review.** Landed across four
commits: `88b0b7d` (Conv/Pool merged with signature defaults already applied,
pre-merge), `00fde43` (freeze + ruff + `transformer.py`), `a7d5ac0`
(`transformer_v2.py` migration), `c50ab0d` (older nodes, docs, changelog) —
then revised after the critical review in
`docs/dev_plans/immutable_objects_pr_review.md` (`579c73b` plus the follow-up
change set). This document records the design choices that diverged from the
original plan (**Design choice (changed from plan)**) and the ones the review
overturned or added (**Design choice (revised after review)**); the review
document holds the full findings with the options weighed for each.

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
        for key, value in config.items():
            _validate_immutable(value, key, type(self).__name__)
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

**Design choice (revised after review): config values are validated, not
trusted.** As first landed, the freeze was **shallow**: attribute writes and
`config` key reassignment were blocked, but a mutable value stored *under* a
key was not — `GaussianEnergy(table=[1, 2, 3])` constructed fine and stayed
mutable through `config["table"]`, and B006 cannot catch it (it flags only
container *literals* as parameter defaults, not mutables inside constructor
calls). The "construct only with immutable scalar config values" contract was
documented but unenforced, so "provably immutable" held only by convention.
Options weighed (review finding 3):

- **Enforce the contract (chosen).** `_validate_immutable` recursively accepts
  `int`, `float`, `str`, `bool`, `bytes`, `None`, and tuples of those; any
  other type raises `TypeError` naming the class and key. Tuples are the
  spelling for structured values — a future per-channel alpha is a tuple, not
  a list or a `jnp` array. Nothing in-repo passed anything else, so no caller
  migration was needed beyond the one violation below.
- **Keep the convention, soften the claim (rejected).** Would preserve room
  for array-valued config, at the cost of the guarantee being only as strong
  as the next subclass author's discipline.

The validation immediately caught one live violation:
`TransformerBlock.initialize_params` built transient
`NormalInitializer(std=1.0 / jnp.sqrt(embed_dim))` — `jnp.sqrt` returns a 0-d
JAX array, so the config value was an `ArrayImpl`. `embed_dim` is a static
Python int, so those five sites now use `math.sqrt`, keeping the value a host
float and avoiding a device op for a compile-time constant.

Notes (confirmed in implementation):
- No concrete subclass writes any instance attribute other than `config`, so no
  subclass `__init__` needed adjustment.
- `__setattr__` is the right freeze mechanism because `copy.copy` /
  `copy.deepcopy` reconstruct via `__dict__` update / `__setstate__`, which
  bypass `__setattr__`. The node finalize path `_with_graph_info` does
  `copy.copy(self)` on the **node** (sharing the activation/energy/init objects
  by reference), so freezing those objects does not interfere with
  copy-on-finalize (`TestCopyOnFinalize` covers this).
- The `FrozenConfig` docstring states the enforced contract: config values are
  validated at construction — immutable scalars and tuples of those only — so
  the whole object is immutable, not just its top level.
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
where `None` is a real value: `weight_init` on weight-free nodes
(`weight_init: Optional[InitializerBase] = None`; the pooling nodes pass
`None` deliberately). `TransformerBlock.internal_activation` was initially
kept `Optional` too, but the review overturned that (see below).
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

**Design choice (revised after review): reject `None` at construction.**
Deleting the body fallbacks also deleted the only runtime `None` guard (the
`ValueError: Node '<name>' has no energy functional configured` in the energy
step) without replacing it, and annotations are not enforced at runtime
(review finding 2). The failure modes were obscure: `Linear(energy=None)`
died as `AttributeError: type object 'NoneType' has no attribute 'energy'` at
energy computation, potentially inside a jitted trace with no node name;
`Linear(weight_init=None)` — previously a silent `NormalInitializer(0, 0.05)`
fallback — died inside parameter initialization. Since `None` lost its
use-the-default meaning, `NodeBase.__init__` now validates at construction,
where the node name is in hand: `activation`, `energy`, and `latent_init`
must be instances of their base classes, else `TypeError` naming the node and
parameter; `weight_init` must be an `InitializerBase` or `None` (weight-free —
the pooling nodes pass it deliberately). The alternative — trust the
annotations and let misuse fail downstream — was rejected for the error
quality. A related choice was accepted knowingly (review finding 4): a custom
node forwarding `**kwargs` without setting `energy` previously hit that
`ValueError`; it now silently receives the `NodeBase` signature defaults
(`GaussianEnergy`, `IdentityActivation`) — the canonical predictive-coding
defaults, consistent with `compute_energy`'s own `GaussianEnergy()` default —
and the changelog names the change.

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

**`transformer.py` (`TransformerBlock`) — design choice (revised after
review).** The v1 transformer's
`internal_activation: Optional[ActivationBase] = None` +
`internal_activation or GeluActivation()` body form was first migrated to a
`GeluActivation()` signature default with `None` kept meaningful: explicit
`None` selected `forward()`'s identity branch, previously unreachable because
the `or` idiom forced a non-`None` value. The review (finding 1) overturned
that: under the old semantics explicit `None` meant GELU, so an external
caller passing `None` would have gotten a **silent numerical change** in the
feed-forward sublayer (GELU → identity) with no error and no changelog notice
— and the API would carry two spellings of identity (`None` and
`IdentityActivation()`), with the `else` branch in `forward()` duplicating
what `IdentityActivation` already is. Options weighed:

- **Require `ActivationBase`, reject `None` (chosen).** The annotation is
  non-`Optional`, `__init__` raises `TypeError` for a non-`ActivationBase`
  value, the identity branch in `forward()` is deleted (one call path:
  `type(a).forward(x, a.config)` — valid for every subclass, since
  `(x, config)` is the `ActivationBase.forward` contract), and identity is
  spelled `IdentityActivation()`. The break is named in the changelog.
- **Keep `None` → identity, normalized in `__init__` (rejected).** Would
  retain a magic value and still silently change meaning for old callers.

Covered by `test_transformer_block_internal_activation_default_and_none`
(shared frozen GELU default; `None` → `TypeError`; identity via
`IdentityActivation()`).

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
- **Design choice (revised after review): lint everything, scope only the
  E402 exception.** As first landed, `extend-exclude` listed `examples`,
  `tests`, and `scripts`, justified by the E402-by-design idiom
  (`set_jax_flags_before_importing_jax` before `import jax`) and "tests are
  not shipped". The review (finding 5) rejected the directory exclusion: the
  E402 justification covers `examples/` and `scripts/` only, and excluding
  `tests/` turns pyflakes off exactly where unused imports accumulate — the
  class of defect this change had already fixed by hand in
  `test_external_custom_node.py`. The standard mechanism is
  `[tool.ruff.lint.per-file-ignores]`: `examples/**` and `scripts/**` ignore
  `E402` only; `tests/` is fully linted. Widening the scope surfaced 53
  findings, all fixed — 45 auto-fixed (unused imports, placeholder-less
  f-strings) and 8 manual (`__all__` for the re-exports in the two `examples`
  `__init__.py` files, `except Exception:` for a bare `except`, four dead
  variable assignments).
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

- **`06_custom_nodes.md`** — Step 1 "Key points": object-valued defaults go in
  the signature (the objects are immutable — a read-only `config` mapping of
  validated immutable values, attribute writes blocked — so the module-level
  singleton cannot leak state between nodes); `None` is not a "use the
  default" sentinel (`NodeBase.__init__` raises `TypeError`; `weight_init=None`
  stays legal for weight-free nodes); body fill only for defaults computed
  from another argument; never default a parameter to a `dict`/`list`/`set`
  literal — `ruff` B006 rejects it in CI.
- **`11_api_activations_and_energy.md`** and **`13_api_initializers.md`**: the
  custom-subclass sections state that instances are frozen after
  `super().__init__(**config)` — all configuration must pass through `config`,
  setting an instance attribute in a subclass `__init__` raises
  `AttributeError` — and (added after review) the value contract: immutable
  scalars and tuples of those only; a list, dict, or array raises `TypeError`.

No guide showed the body-fill constructor pattern. One example did:
`examples/resnet18_cifar10_demo.py` filled `activation`/`output_weight_init`
in function bodies (`if activation is None: activation = ReLUActivation()`)
without deriving them from another argument — the pattern the guide bans —
and, being in `examples/`, nothing would ever flag it. It was migrated to
signature defaults after the review (finding 7).

### 6. Changelog (added beyond plan, expanded after review)

`CHANGELOG.md`'s Unreleased entry covers: base classes frozen at construction;
config values validated (scalars and tuples only, `TypeError` otherwise);
`None` guards removed and defaults migrated from method bodies to
constructors; `Optional` dropped from always-required constructor arguments;
two breaking notices — `None` rejected for `activation`/`energy`/`latent_init`
(`weight_init=None` still means weight-free) and
`TransformerBlock(internal_activation=None)` no longer meaning GELU; the
custom-node silent-default change (review finding 4); and ruff added to
pre-commit for linting only — the first draft said "linting and formatting",
contradicting the hook comment (formatting stays with Black; review
finding 6).

## Critical files — as touched

- `fabricpc/core/_frozen.py` — new `FrozenConfig` mixin (single freeze
  source); after review, recursive scalar/tuple validation of config values.
- `fabricpc/core/activations.py`, `energy.py`, `initializers.py` — bases
  inherit `FrozenConfig`; docstring freeze notes; `compute_energy` /
  `compute_energy_gradient` signature defaults.
- `fabricpc/core/types.py` — `NodeInfo` field comments state the
  never-`None` invariants.
- `fabricpc/nodes/convolutional.py`, `pooling.py` — signature defaults (landed
  pre-merge in the Conv/Pool PR, `88b0b7d`); `ConvNode` bias fail-fast guard.
- `fabricpc/nodes/transformer_v2.py` — 5 nodes, `or`-form → signature defaults.
- `fabricpc/nodes/transformer.py` — `internal_activation` signature default;
  after review, `None` rejected at construction, single `forward()` call path,
  and `math.sqrt` (host float) instead of `jnp.sqrt` (0-d array) for the
  transient initializer stds.
- `fabricpc/nodes/linear.py`, `identity.py`, `skip_connection.py`,
  `linear_residual.py`, `linear_explicit_grad.py`, `storkey_hopfield.py` —
  `Optional` dropped; hidden `initialize_params` re-defaults removed.
- `pyproject.toml`, `.pre-commit-config.yaml` — ruff B006 (not B008), version
  pinned to the hook rev; after review, full-tree lint scope with
  `per-file-ignores` limiting the exception to E402 in `examples/**` and
  `scripts/**`.
- `fabricpc/nodes/base.py` — module and class docstrings state the convention;
  after review, construction-time `TypeError` validation naming the node.
- `docs/user_guides/06_custom_nodes.md`, `11_api_activations_and_energy.md`,
  `13_api_initializers.md` — convention, freeze, and (after review) value
  contract and `None`-rejection notes.
- `examples/resnet18_cifar10_demo.py` — body-fill defaults migrated to
  signature defaults (after review).
- `CHANGELOG.md` — Unreleased entry.
- `tests/test_immutable_config.py` — new test module (see Verification).
- Assorted `tests/` and `examples/` files — lint fixes surfaced by the widened
  ruff scope.

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
   (`test_transformer_block_internal_activation_default_and_none`, revised
   after review): the default is one shared frozen `GeluActivation` singleton;
   explicit `internal_activation=None` raises `TypeError`; identity is
   requested with `IdentityActivation()`.
7. **Config value validation** (`TestConfigValueValidation`, added after
   review): scalars, tuples, and nested tuples accepted; list, dict, set,
   list-inside-tuple, and `jnp` array each raise `TypeError`; a concrete
   subclass (`GaussianEnergy(precision=[1.0, 2.0])`) routes through the same
   validation.
8. **Constructor validation** (`TestNodeConstructorValidation`, added after
   review): `None` for `activation`/`energy`/`latent_init` raises `TypeError`
   naming the node; a non-initializer `weight_init` is rejected;
   `weight_init=None` on a pooling node stays legal.

Plus the planned non-test checks: full `pytest` suite (262 passed, 1
pre-existing multi-GPU skip after the review changes); `ruff check .` clean
over the widened scope (zero B006 findings) and `pre-commit run --all-files`
confirming the hook wiring; the custom-subclass snippets in guides 06/11/13
construct cleanly against the frozen bases (each calls only
`super().__init__(**config)`).

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

The critical review (`docs/dev_plans/immutable_objects_pr_review.md`) then
hardened the result in four ways: the immutability claim was made enforceable
(config values validated — scalars and tuples only), `None` was removed as an
accepted-but-unhandled value (constructor `TypeError` naming the node,
including `TransformerBlock.internal_activation`, whose `None` → identity
semantics were reverted before any release shipped them), the ruff scope was
widened from `fabricpc/`-only to the full tree with a per-file E402 exception,
and the one example still using the banned body-fill idiom was migrated. Two
review findings were accepted as-is and documented rather than changed: custom
nodes forwarding `**kwargs` now silently receive the canonical defaults
instead of raising, and the freeze bars external subclasses from caching state
on `self` — config is the single home for state, consistent with the
all-static-methods design.
