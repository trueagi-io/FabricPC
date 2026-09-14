"""
Executes the calls ``docs/user_guides/17_training_with_epc.md`` makes, on a
small synthetic graph, so the printed regime formats, the ``evaluate``
indexing, the probe's CSV, and the ePC-then-sPC composition are guarded by
something that runs. ``tests/test_doc_snippets.py`` parses the guide's
blocks and checks their signatures; it executes nothing but imports.
"""

import re

import jax
import numpy as np
import optax
import pytest

from conftest import ListLoader
from fabricpc.core import EPCInference, epsilon_spectrum
from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy, graph_energy
from fabricpc.core.inference import InferenceSGD, InferenceSchedule
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import (
    RegimeProbe,
    build_clamps,
    evaluate,
    read_regime_csv,
    train,
)
from fabricpc.utils.dashboarding import make_inference_history

BATCH, N_BATCHES, EPOCHS = 8, 3, 2
BANDS = {
    "backprop-like",
    "partially relaxed",
    "near PC equilibrium",
    "no positive curvature",
}
UNSTABLE = re.compile(r"^unstable: eta\*lambda_max = [0-9.e+-]+ > 2$")
INDEFINITE = re.compile(
    r"^indefinite: negative curvature carrying \d+% of the gradient grows "
)
BAND = re.compile(
    r"^eta\*T\*lambda_max = [0-9.e+-]+ \(gradient-weighted relaxed fraction -?[0-9.]+, "
    r"fastest mode -?[0-9.]+\): (backprop-like|partially relaxed|near PC equilibrium|no positive curvature)"
    r"(; output-layer gradient reverses on the top mode \(eta\*\(lambda_max - 1\) = [0-9.e+-]+\))?$"
)


def build(inference):
    """x(6) -> h1(5, tanh) -> h2(5, tanh) -> y(3, softmax + cross-entropy)."""
    w_init = NormalInitializer(std=0.5)
    x = IdentityNode(shape=(6,), name="x")
    h1 = Linear(shape=(5,), name="h1", activation=TanhActivation(), weight_init=w_init)
    h2 = Linear(shape=(5,), name="h2", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(3,),
        name="y",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=w_init,
    )
    return graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference,
    )


def batches(key):
    out = []
    for i in range(N_BATCHES):
        k = jax.random.fold_in(key, i)
        out.append(
            {
                "x": jax.random.normal(k, (BATCH, 6)),
                "y": jax.random.randint(jax.random.fold_in(k, 1), (BATCH,), 0, 3),
            }
        )
    return out


def spectrum_at(structure, params, clamps, key):
    state = initialize_graph_state(structure, BATCH, key, clamps=clamps, params=params)
    return epsilon_spectrum(params, state, clamps, structure)


@pytest.fixture
def setup(rng_key):
    structure = build(EPCInference(eta_infer=0.01, infer_steps=5))  # provisional
    params = initialize_params(structure, rng_key)
    data = batches(rng_key)
    clamps = build_clamps(data[0], structure, clamp_target=True)
    spectrum = spectrum_at(structure, params, clamps, jax.random.PRNGKey(1))
    return structure, params, data, clamps, spectrum


class TestSpectrumAndGrid:
    def test_spectrum_fields(self, setup):
        _, _, _, _, spectrum = setup
        lam_max = float(spectrum.lambda_max)
        assert lam_max > 0 and np.isfinite(float(spectrum.lambda_min))
        assert 0.0 <= float(spectrum.negative_weight) <= 1.0

    def test_grid_bands_and_formats(self, setup):
        """Every cell of the tutorial's grid reads a known band, the label
        matches one of the three ``str(regime)`` forms, and the fraction of
        the bound decides ``unstable`` exactly."""
        _, _, _, _, spectrum = setup
        lam_max = float(spectrum.lambda_max)
        for c in [0.01, 0.03, 0.1, 0.3, 0.9, 1.5]:
            eta = c * 2.0 / lam_max
            for T in [1, 2, 5, 10, 20, 50]:
                r = EPCInference(eta_infer=eta, infer_steps=T).regime(spectrum)
                assert r.band in BANDS
                assert r.unstable == (c > 1.0)
                text = str(r)
                assert (
                    UNSTABLE.match(text) or INDEFINITE.match(text) or BAND.match(text)
                ), text
                if r.unstable:
                    assert UNSTABLE.match(text)
                if T % 2 == 0:
                    assert not r.output_gradient_reverses
        # f̄ rises with T at fixed eta below the bound
        f = [
            EPCInference(eta_infer=0.1 * 2 / lam_max, infer_steps=T)
            .regime(spectrum)
            .f_weighted
            for T in (1, 5, 20)
        ]
        assert f[0] < f[1] < f[2]


class TestProbeWorkflow:
    def test_probe_with_evaluate_accuracy_and_csv(self, setup, rng_key, tmp_path):
        structure, params, data, clamps, spectrum = setup
        lam_max = float(spectrum.lambda_max)
        eta, T = round(0.3 * 2.0 / lam_max, 4), 5
        structure = build(EPCInference(eta_infer=eta, infer_steps=T))
        params = initialize_params(structure, rng_key)
        loader = ListLoader(data)
        csv_path = tmp_path / "epc_regime_track.csv"
        probe = RegimeProbe(
            structure, clamps, every=1, iters=12, key=rng_key, csv_path=csv_path
        )

        def on_epoch(ctx):
            accuracy = evaluate(
                ctx.params, ctx.structure, loader, ctx.config, ctx.epoch_key
            )["accuracy"]
            probe.on_epoch(ctx, accuracy)

        result = train(
            params,
            structure,
            loader,
            optax.adamw(1e-3, weight_decay=0.1),
            {"num_epochs": EPOCHS},
            rng_key,
            verbose=False,
            iter_callback=probe.on_iter,
            epoch_callback=on_epoch,
        )
        assert len(probe.probe_rows()) == EPOCHS * N_BATCHES
        for row in probe.epoch_rows():
            assert (
                isinstance(row["test_accuracy"], float)
                and 0.0 <= row["test_accuracy"] <= 1.0
            )
        text = probe.summary(chance=1 / 3)
        assert (
            "stability bound eta*lambda_max > 2:" in text
            and "lambda_max per epoch" in text
        )
        assert probe.first_crossing() is None or isinstance(
            probe.first_crossing(), tuple
        )
        phases = probe.growth_phases()
        assert [e for e, _, _ in phases] == list(range(EPOCHS))
        metadata, rows = read_regime_csv(csv_path)
        assert (
            metadata["eta_infer"] == eta
            and metadata["infer_steps"] == T
            and metadata["every"] == 1
        )
        assert len(rows) == len(probe.rows)
        end = structure.config["inference"].regime(
            spectrum_at(structure, result.params, clamps, jax.random.PRNGKey(1))
        )
        assert (
            UNSTABLE.match(str(end))
            or INDEFINITE.match(str(end))
            or BAND.match(str(end))
        )

    def test_restart_from_saved_state(self, setup, rng_key):
        """The restart remedy: continue from ``result.params`` and
        ``result.opt_state`` at a lower eta with ``start_epoch``."""
        structure, params, data, clamps, spectrum = setup
        loader = ListLoader(data)
        first = train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            verbose=False,
        )
        lower = build(
            EPCInference(
                eta_infer=0.5
                * float(structure.config["inference"].config["eta_infer"]),
                infer_steps=5,
            )
        )
        probe = RegimeProbe(lower, clamps, every=1, iters=12, key=rng_key)
        second = train(
            first.params,
            lower,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            opt_state=first.opt_state,
            start_epoch=1,
            verbose=False,
            iter_callback=probe.on_iter,
            epoch_callback=lambda ctx: probe.on_epoch(ctx, 0.5),
        )
        assert second.step == N_BATCHES
        assert [r["epoch"] for r in probe.probe_rows()] == [1] * N_BATCHES


class TestComposition:
    def test_schedule_probe_and_energy(self, setup, rng_key):
        structure, params, data, clamps, spectrum = setup
        lam_max = float(spectrum.lambda_max)
        T_epc, T_spc = 5, 10
        epc = EPCInference(eta_infer=round(0.03 * 2.0 / lam_max, 4), infer_steps=T_epc)
        schedule = build(
            InferenceSchedule(epc, InferenceSGD(eta_infer=0.1, infer_steps=T_spc))
        )
        sched_params = initialize_params(schedule, rng_key)

        history = make_inference_history(schedule)
        state0 = initialize_graph_state(
            schedule, BATCH, jax.random.PRNGKey(1), clamps=clamps, params=sched_params
        )
        _, states = history(sched_params, state0, clamps)

        def energy_at(i):
            return float(
                graph_energy(jax.tree_util.tree_map(lambda a: a[i], states), schedule)
            )

        assert states.nodes["y"].energy.shape[0] == T_epc + T_spc + 1
        assert energy_at(T_epc) < energy_at(0)
        assert energy_at(T_epc + T_spc) < energy_at(T_epc)

        # the probe under a schedule: blank without inference=, filled with it
        blank = RegimeProbe(schedule, clamps, every=1, iters=12, key=rng_key)
        assert blank.inference is None
        probe = RegimeProbe(
            schedule, clamps, every=1, iters=12, inference=epc, key=rng_key
        )
        loader = ListLoader(data)
        train(
            sched_params,
            schedule,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            verbose=False,
            iter_callback=probe.on_iter,
            epoch_callback=lambda ctx: probe.on_epoch(
                ctx,
                evaluate(ctx.params, ctx.structure, loader, ctx.config, ctx.epoch_key)[
                    "accuracy"
                ],
            ),
        )
        assert probe.metadata["infer_steps"] == T_epc
        assert all(row["f_weighted"] is not None for row in probe.probe_rows())
        assert len(probe.probe_rows()) == N_BATCHES
        assert isinstance(probe.epoch_rows()[0]["test_accuracy"], float)
