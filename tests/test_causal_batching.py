import numpy as np

from fabricpc.continual.causal import CausalSupportFeatureBuilder
from fabricpc.continual.config import SupportConfig
from fabricpc.continual.support import SupportManager


class StubBank:
    def __init__(self, mean_gain: np.ndarray, confidence: np.ndarray):
        self._mean_gain = mean_gain
        self._confidence = confidence

    def mean_gain(self) -> np.ndarray:
        return self._mean_gain

    def column_confidence(self, _target: float) -> np.ndarray:
        return self._confidence


def test_build_features_batch_matches_scalar_builder():
    builder = CausalSupportFeatureBuilder.from_config(
        num_columns=8,
        num_tasks=5,
        topk_nonshared=4,
        reserve_indices=[6],
    )

    base_z = np.linspace(-1.0, 1.0, 8)
    cert_general = np.linspace(0.1, 0.8, 8)
    cert_specific = np.linspace(0.2, 0.9, 8)
    cert_demotion = np.linspace(-0.3, 0.4, 8)
    cert_saturation = np.linspace(0.0, 0.7, 8)
    novelty = np.linspace(-0.5, 0.2, 8)
    saturation = np.linspace(0.3, 1.0, 8)
    recent_penalty = np.linspace(0.0, 0.35, 8)
    reserve_bonus = np.linspace(0.4, 1.1, 8)
    fingerprint_mean = np.arange(40, dtype=np.float64).reshape(8, 5) / 10.0
    fingerprint_confidence = np.linspace(0.1, 0.8, 8)

    indices = [2, 5, 6]
    roles = ["challenger", "reuse", "diverse"]
    chosen_sets = [(0, 1, 3), (0, 2), (1, 2, 4)]
    current_task_id = 2

    def struct_sim(i: int, j: int) -> float:
        return 1.0 if i == j else 0.0

    def causal_sim(i: int, j: int) -> float:
        vi = fingerprint_mean[i]
        vj = fingerprint_mean[j]
        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        if ni < 1e-6 or nj < 1e-6:
            return 0.0
        return float(np.dot(vi, vj) / (ni * nj))

    scalar = np.stack(
        [
            builder.build_feature(
                idx=idx,
                role=role,
                chosen=chosen,
                base_z=base_z,
                cert_general=cert_general,
                cert_specific=cert_specific,
                cert_demotion=cert_demotion,
                cert_saturation=cert_saturation,
                novelty=novelty,
                saturation=saturation,
                recent_penalty=recent_penalty,
                reserve_bonus=reserve_bonus,
                fingerprint_mean=fingerprint_mean,
                fingerprint_confidence=fingerprint_confidence,
                struct_similarity_fn=struct_sim,
                causal_similarity_fn=causal_sim,
                current_task_id=current_task_id,
            )
            for idx, role, chosen in zip(indices, roles, chosen_sets)
        ],
        axis=0,
    )

    batched = builder.build_features_batch(
        indices=indices,
        roles=roles,
        chosen_sets=chosen_sets,
        base_z=base_z,
        cert_general=cert_general,
        cert_specific=cert_specific,
        cert_demotion=cert_demotion,
        cert_saturation=cert_saturation,
        novelty=novelty,
        saturation=saturation,
        recent_penalty=recent_penalty,
        reserve_bonus=reserve_bonus,
        fingerprint_mean=fingerprint_mean,
        fingerprint_confidence=fingerprint_confidence,
        current_task_id=current_task_id,
    )

    assert batched.shape == scalar.shape
    np.testing.assert_allclose(batched, scalar, rtol=1e-6, atol=1e-6)


def test_support_manager_causal_guidance_uses_batched_scoring():
    config = SupportConfig(
        causal_max_effective_scale=0.5,
        causal_similarity_conf_target=1.0,
    )
    manager = SupportManager(
        num_columns=8,
        num_shared=2,
        topk_nonshared=2,
        config=config,
        num_tasks=5,
    )

    predictor = manager.causal_predictor
    assert predictor is not None
    predictor.trained = True
    predictor.input_dim = 21
    predictor.mean_ = np.zeros(21, dtype=np.float64)
    predictor.std_ = np.ones(21, dtype=np.float64)
    predictor.beta_ = np.zeros(22, dtype=np.float64)
    predictor.beta_[11] = 1.0  # weight only fp_cur

    manager.causal_trust.last_diag["mix_gate"] = 0.5
    fingerprint_mean = np.zeros((8, 5), dtype=np.float64)
    fingerprint_mean[2, 1] = 0.05
    fingerprint_mean[3, 1] = 0.06
    fingerprint_mean[4, 1] = 0.50
    manager.causal_bank = StubBank(
        mean_gain=fingerprint_mean,
        confidence=np.ones(8, dtype=np.float64),
    )

    selected = manager._apply_causal_guidance(task_id=1, initial_selected=[2, 3])

    assert selected == (4, 3)
