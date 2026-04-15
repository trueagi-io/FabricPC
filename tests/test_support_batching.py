from types import SimpleNamespace

import numpy as np

import fabricpc.continual.trainer as trainer_module
from fabricpc.continual.causal import AgreementTracker
from fabricpc.continual.support import (
    DemotionBank,
    HybridSelectorPolicy,
    ReplayBuffer,
    SupportBank,
)


def test_support_bank_mean_accuracy_by_column():
    bank = SupportBank()
    bank.add(task_id=0, support_cols=[2, 3], accuracy=0.8, loss=0.2)
    bank.add(task_id=1, support_cols=[3, 4], accuracy=0.6, loss=0.3)

    mean_acc = bank.get_mean_accuracy_by_column(num_columns=6)

    expected = np.zeros(6)
    expected[2] = 0.8
    expected[3] = 0.7
    expected[4] = 0.6
    np.testing.assert_allclose(mean_acc, expected)


def test_hybrid_selector_policy_batched_scoring():
    policy = HybridSelectorPolicy(
        num_columns=8,
        num_shared=2,
        topk_nonshared=2,
    )
    policy.column_scores[2:] = np.array([0.1, 0.2, 0.9, 0.4, 0.5, 0.3])
    policy.task_preferences[1] = np.zeros(8)
    policy.task_preferences[1][2:] = np.array([0.0, 0.5, 0.0, 0.1, 0.0, 0.0])

    support_bank = SupportBank()
    support_bank.add(task_id=0, support_cols=[3, 5], accuracy=0.9, loss=0.1)
    support_bank.add(task_id=1, support_cols=[5, 6], accuracy=0.7, loss=0.2)

    demotion_bank = DemotionBank()
    demotion_bank.add_demotion(
        task_id=1,
        column_idx=7,
        reason="test",
        score_before=0.0,
        score_after=0.0,
    )

    selected = policy.select_support(
        task_id=1,
        support_bank=support_bank,
        demotion_bank=demotion_bank,
        exploration_rate=0.0,
    )

    assert selected == (3, 4)


def test_support_swap_audit_vectorized_generation(monkeypatch):
    trainer = trainer_module.SequentialTrainer.__new__(trainer_module.SequentialTrainer)
    trainer.config = SimpleNamespace(
        audit=SimpleNamespace(
            support_swap_audit_enable=True,
            support_swap_audit_max_swaps=3,
            support_audit_max_batches=1,
            support_audit_current_weight=1.0,
            support_audit_old_weight=0.5,
        ),
        columns=SimpleNamespace(
            shared_columns=2,
            num_columns=6,
        ),
        training=SimpleNamespace(training_mode="pc"),
    )
    trainer.support_manager = SimpleNamespace(
        support_bank=SupportBank(),
        current_state=SimpleNamespace(active_nonshared=(2, 3)),
    )
    trainer.params = None
    trainer.structure = None
    trainer.rng_key = np.array([0, 1], dtype=np.uint32)

    trainer.support_manager.support_bank.add(
        task_id=0, support_cols=[2], accuracy=0.4, loss=0.1
    )
    trainer.support_manager.support_bank.add(
        task_id=0, support_cols=[3], accuracy=0.6, loss=0.1
    )
    trainer.support_manager.support_bank.add(
        task_id=1, support_cols=[4], accuracy=0.9, loss=0.1
    )
    trainer.support_manager.support_bank.add(
        task_id=1, support_cols=[5], accuracy=0.7, loss=0.1
    )

    eval_calls = []

    def fake_evaluate_pcn(_params, _structure, loader, _config, _eval_key):
        eval_calls.append(len(loader))
        return {"loss": 1.0}

    monkeypatch.setattr(trainer_module, "evaluate_pcn", fake_evaluate_pcn)

    current_task = SimpleNamespace(
        task_id=2,
        test_loader=[
            (np.zeros((2, 1)), np.zeros((2, 1))),
            (np.zeros((3, 1)), np.zeros((3, 1))),
        ],
    )
    old_task = SimpleNamespace(
        task_id=1,
        test_loader=[
            (np.zeros((2, 1)), np.zeros((2, 1))),
            (np.zeros((4, 1)), np.zeros((4, 1))),
        ],
    )

    np.random.seed(0)
    rows = trainer_module.SequentialTrainer._run_support_swap_audit(
        trainer,
        current_task_data=current_task,
        old_task_data=old_task,
        verbose=False,
    )

    assert len(rows) == 3
    assert eval_calls == [1, 1]
    assert [(row["swap_out"], row["swap_in"]) for row in rows] == [
        (2, 4),
        (2, 5),
        (3, 4),
    ]
    for row in rows:
        assert row["chosen_current_loss"] == 1.0
        assert row["chosen_old_loss"] == 1.0
        np.testing.assert_allclose(row["old_gain"], row["current_gain"] * 0.5)
        np.testing.assert_allclose(
            row["combined_gain"], row["current_gain"] + 0.5 * row["old_gain"]
        )


def test_replay_buffer_sample_without_global_concatenation_regression():
    buffer = ReplayBuffer(max_samples_per_task=10, max_total_samples=20)
    images_a = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels_a = np.arange(6, dtype=np.int32)
    images_b = np.arange(20, 32, dtype=np.float32).reshape(6, 2)
    labels_b = np.arange(10, 16, dtype=np.int32)

    buffer.add_task_samples(0, images_a, labels_a, replace=True)
    buffer.add_task_samples(1, images_b, labels_b, replace=True)

    np.random.seed(0)
    sample = buffer.sample(batch_size=5, exclude_task=1)
    assert sample is not None
    sampled_images, sampled_labels = sample
    assert sampled_images.shape[0] == 5
    assert sampled_labels.shape[0] == 5
    assert np.all(np.isin(sampled_labels, labels_a))


def test_replay_buffer_sample_batches_returns_dense_epoch_batches():
    buffer = ReplayBuffer(max_samples_per_task=10, max_total_samples=20)
    images_a = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels_a = np.arange(6, dtype=np.int32)
    images_b = np.arange(20, 32, dtype=np.float32).reshape(6, 2)
    labels_b = np.arange(10, 16, dtype=np.int32)

    buffer.add_task_samples(0, images_a, labels_a, replace=True)
    buffer.add_task_samples(1, images_b, labels_b, replace=True)

    np.random.seed(0)
    sample = buffer.sample_batches(num_batches=3, batch_size=4, exclude_task=1)
    assert sample is not None
    sampled_images, sampled_labels = sample

    assert sampled_images.shape == (3, 4, 2)
    assert sampled_labels.shape == (3, 4)
    assert np.all(np.isin(sampled_labels, labels_a))


def test_interleaved_loader_presamples_replay_once_per_epoch():
    current_batches = [
        (
            np.full((2, 3), fill_value=1.0, dtype=np.float32),
            np.full((2, 2), fill_value=1.0, dtype=np.float32),
        ),
        (
            np.full((2, 3), fill_value=2.0, dtype=np.float32),
            np.full((2, 2), fill_value=2.0, dtype=np.float32),
        ),
    ]

    class MockLoader:
        def __init__(self, batches):
            self._batches = batches
            self.batch_size = len(batches[0][0])

        def __len__(self):
            return len(self._batches)

        def __iter__(self):
            return iter(self._batches)

    class MockReplayBuffer:
        def __init__(self):
            self.sample_batches_calls = []

        def __len__(self):
            return 1

        def get_task_ids(self):
            return [0]

        def sample_batches(self, num_batches, batch_size, exclude_task=None):
            self.sample_batches_calls.append((num_batches, batch_size, exclude_task))
            replay_images = np.full(
                (num_batches, batch_size, 3), fill_value=9.0, dtype=np.float32
            )
            replay_labels = np.full(
                (num_batches, batch_size, 2), fill_value=9.0, dtype=np.float32
            )
            return replay_images, replay_labels

    loader = trainer_module.InterleavedLoader(
        current_loader=MockLoader(current_batches),
        replay_buffer=MockReplayBuffer(),
        current_task_id=1,
        replay_ratio=0.5,
    )

    mixed_batches = list(loader)

    assert len(mixed_batches) == 2
    assert loader.replay_buffer.sample_batches_calls == [(2, 1, 1)]
    for images, labels in mixed_batches:
        assert images.shape == (3, 3)
        assert labels.shape == (3, 2)
        assert np.sum(np.all(images == 9.0, axis=1)) == 1
        assert np.sum(np.all(labels == 9.0, axis=1)) == 1


def test_agreement_tracker_array_backed_matching_and_state_roundtrip():
    tracker = AgreementTracker(max_history=6)

    for task_id, col_idx, pred, gain in [
        (0, 2, 0.2, 0.1),
        (0, 3, 0.5, 0.4),
        (1, 2, 0.7, 0.6),
        (1, 4, 0.1, 0.2),
    ]:
        tracker.record_prediction(task_id, col_idx, pred, "challenger")
        tracker.record_outcome(task_id, col_idx, gain)

    agreement, matched = tracker.compute_recent_agreement(window=10)
    assert matched == 4
    assert 0.5 <= agreement <= 1.0
    assert len(tracker.predictions) == 4
    assert len(tracker.outcomes) == 4

    state = tracker.save_state()
    restored = AgreementTracker(max_history=6)
    restored.load_state(state)
    restored_agreement, restored_matched = restored.compute_recent_agreement(window=10)
    assert restored_matched == matched
    np.testing.assert_allclose(restored_agreement, agreement)
