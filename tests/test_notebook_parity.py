from fabricpc.continual.parity import (
    DEFAULT_TOLERANCES,
    ParityMetrics,
    compare_against_baseline,
    make_parity_config,
)


def test_make_parity_config_v18_like_mutes_later_mechanisms():
    cfg = make_parity_config("v18_like", seed=7)
    assert cfg.num_tasks == 3
    assert cfg.support.use_replay is False
    assert cfg.support.causal_max_effective_scale == 0.0
    assert cfg.composer_transweave.enable is False
    assert cfg.shell_demotion_transweave.enable is False
    assert cfg.per_weight_causal.enable is False


def test_make_parity_config_v20_2b_like_enables_later_mechanisms():
    cfg = make_parity_config("v20_2b_like", seed=7)
    assert cfg.num_tasks == 3
    assert cfg.support.use_replay is True
    assert cfg.support.causal_max_effective_scale > 0.0
    assert cfg.composer_transweave.enable is True
    assert cfg.shell_demotion_transweave.enable is True
    assert cfg.per_weight_causal.enable is True


def test_compare_against_baseline_pass_and_fail():
    observed = ParityMetrics(
        profile="v20_2b_like",
        seed=42,
        num_tasks=3,
        final_mean_accuracy=0.95,
        average_forgetting=0.05,
        support_diversity=2,
        mean_test_accuracy=0.96,
        mean_causal_examples=16.0,
        mean_causal_mix_gate=0.25,
        mean_transweave_sources=1.5,
        total_training_time_s=12.0,
    )
    baseline = {
        "final_mean_accuracy": 0.94,
        "average_forgetting": 0.06,
        "support_diversity": 2,
        "mean_test_accuracy": 0.95,
        "mean_causal_examples": 15.0,
        "mean_causal_mix_gate": 0.22,
        "mean_transweave_sources": 1.0,
    }

    passing = compare_against_baseline(observed, baseline)
    assert passing.passed

    strict_tolerances = dict(DEFAULT_TOLERANCES)
    strict_tolerances["final_mean_accuracy"] = type(
        DEFAULT_TOLERANCES["final_mean_accuracy"]
    )(abs_tol=0.001)
    failing = compare_against_baseline(observed, baseline, strict_tolerances)
    assert not failing.passed
