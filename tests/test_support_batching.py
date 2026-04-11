from types import SimpleNamespace

import numpy as np

import fabricpc.continual.trainer as trainer_module
from fabricpc.continual.support import (
    DemotionBank,
    HybridSelectorPolicy,
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
