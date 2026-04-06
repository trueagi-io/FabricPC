"""
Unit tests for TransWeave multi-level transfer learning.

Tests the core components:
- sinkhorn_transport: Optimal transport computation
- ComposerTransWeave: Composer-level transfer
- ShellDemotionTransWeave: Within-column shell demotion
- TransWeaveManager: Unified manager
"""

import numpy as np
import pytest

from fabricpc.continual.config import (
    ComposerTransWeaveConfig,
    ShellDemotionTransWeaveConfig,
)
from fabricpc.continual.transweave import (
    sinkhorn_transport,
    cosine_cost_matrix,
    euclidean_cost_matrix,
    ComposerTransWeave,
    ShellDemotionTransWeave,
    TransWeaveManager,
)


class TestSinkhornTransport:
    """Test Sinkhorn optimal transport."""

    def test_uniform_transport(self):
        """Test transport with uniform cost."""
        cost = np.ones((4, 4))
        plan = sinkhorn_transport(cost, eps=0.1, iters=20)

        assert plan.shape == (4, 4)
        # Should be approximately doubly stochastic
        row_sums = plan.sum(axis=1)
        col_sums = plan.sum(axis=0)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4) / 4, decimal=2)
        np.testing.assert_array_almost_equal(col_sums, np.ones(4) / 4, decimal=2)

    def test_diagonal_cost(self):
        """Test transport with diagonal cost structure."""
        # High cost off-diagonal
        cost = np.ones((3, 3)) * 10.0
        np.fill_diagonal(cost, 0.0)

        plan = sinkhorn_transport(cost, eps=0.05, iters=30)

        # Diagonal should dominate
        assert plan[0, 0] > plan[0, 1]
        assert plan[1, 1] > plan[1, 0]
        assert plan[2, 2] > plan[2, 1]

    def test_identity_bonus(self):
        """Test identity bonus parameter."""
        cost = np.ones((4, 4))
        plan_no_bonus = sinkhorn_transport(cost, eps=0.1, iters=20, identity_bonus=0.0)
        plan_with_bonus = sinkhorn_transport(
            cost, eps=0.1, iters=20, identity_bonus=0.5
        )

        # With identity bonus, diagonal should be higher
        diag_no_bonus = np.diag(plan_no_bonus).sum()
        diag_with_bonus = np.diag(plan_with_bonus).sum()
        assert diag_with_bonus > diag_no_bonus

    def test_custom_weights(self):
        """Test transport with custom marginal weights."""
        cost = np.ones((3, 3))
        source_weights = np.array([0.5, 0.3, 0.2])
        target_weights = np.array([0.2, 0.5, 0.3])

        plan = sinkhorn_transport(
            cost,
            source_weights=source_weights,
            target_weights=target_weights,
            eps=0.1,
            iters=30,
        )

        # Row sums should approximately match source weights
        row_sums = plan.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, source_weights, decimal=1)


class TestCostMatrices:
    """Test cost matrix computations."""

    def test_cosine_cost_identical(self):
        """Test cosine cost for identical vectors."""
        source = np.array([[1.0, 0.0], [0.0, 1.0]])
        target = source.copy()

        cost = cosine_cost_matrix(source, target)

        # Identical vectors should have cost 0
        assert cost[0, 0] < 0.01
        assert cost[1, 1] < 0.01
        # Orthogonal vectors should have cost 1
        assert np.abs(cost[0, 1] - 1.0) < 0.01
        assert np.abs(cost[1, 0] - 1.0) < 0.01

    def test_euclidean_cost(self):
        """Test Euclidean distance cost."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]])
        target = np.array([[0.0, 0.0], [0.0, 1.0]])

        cost = euclidean_cost_matrix(source, target)

        # Same point should have cost 0
        assert cost[0, 0] < 0.01
        # Distance between (1,0) and (0,1) should be 2 (squared)
        assert np.abs(cost[1, 1] - 2.0) < 0.01


class TestComposerTransWeave:
    """Test Composer-level TransWeave."""

    def test_init(self):
        """Test initialization."""
        config = ComposerTransWeaveConfig()
        composer_tw = ComposerTransWeave(config)
        assert len(composer_tw.task_representations) == 0

    def test_register_task(self):
        """Test task registration."""
        config = ComposerTransWeaveConfig()
        composer_tw = ComposerTransWeave(config)

        # Register a task
        num_heads = 2
        num_cols = 4
        hidden_dim = 8
        key_dim = 4

        attention = np.random.randn(num_heads, num_cols, num_cols)
        query = np.random.randn(num_heads, hidden_dim, key_dim)
        key = np.random.randn(num_heads, hidden_dim, key_dim)
        value = np.random.randn(num_heads, hidden_dim, key_dim)
        output = np.random.randn(num_heads * key_dim, hidden_dim)

        composer_tw.register_task(
            task_id=0,
            attention_weights=attention,
            query_projections=query,
            key_projections=key,
            value_projections=value,
            output_projection=output,
        )

        assert 0 in composer_tw.task_representations
        assert composer_tw.task_representations[0].attention_weights.shape == (
            num_heads,
            num_cols,
            num_cols,
        )

    def test_compute_transfer_warmup(self):
        """Test transfer during warmup period."""
        config = ComposerTransWeaveConfig(warmup_tasks=2)
        composer_tw = ComposerTransWeave(config)

        # Only register 1 task (below warmup)
        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4
        attention = np.random.randn(num_heads, num_cols, num_cols)
        query = np.random.randn(num_heads, hidden_dim, key_dim)
        key = np.random.randn(num_heads, hidden_dim, key_dim)
        value = np.random.randn(num_heads, hidden_dim, key_dim)

        composer_tw.register_task(
            task_id=0,
            attention_weights=attention,
            query_projections=query,
            key_projections=key,
            value_projections=value,
            output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
        )

        # Compute transfer for task 1 (should be in warmup)
        result = composer_tw.compute_transfer(
            target_task_id=1,
            current_attention=attention,
            current_queries=query,
            current_keys=key,
            current_values=value,
        )

        # Should return unmodified during warmup
        assert result.diagnostics.get("warmup", False)
        assert result.transfer_strength == 0.0

    def test_compute_transfer_after_warmup(self):
        """Test transfer after warmup period."""
        config = ComposerTransWeaveConfig(warmup_tasks=1, transfer_strength=0.3)
        composer_tw = ComposerTransWeave(config)

        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4

        # Register 2 tasks
        for task_id in range(2):
            attention = np.random.randn(num_heads, num_cols, num_cols) * 0.1
            attention = np.abs(attention)  # Make positive
            attention = attention / attention.sum(axis=2, keepdims=True)  # Normalize

            composer_tw.register_task(
                task_id=task_id,
                attention_weights=attention,
                query_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                key_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                value_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
            )

        # Compute transfer for task 2
        current_attention = np.random.randn(num_heads, num_cols, num_cols)
        current_attention = np.abs(current_attention)
        current_attention = current_attention / current_attention.sum(
            axis=2, keepdims=True
        )

        result = composer_tw.compute_transfer(
            target_task_id=2,
            current_attention=current_attention,
            current_queries=np.random.randn(num_heads, hidden_dim, key_dim),
            current_keys=np.random.randn(num_heads, hidden_dim, key_dim),
            current_values=np.random.randn(num_heads, hidden_dim, key_dim),
        )

        # Should have transfer from previous tasks
        assert len(result.source_tasks) > 0
        assert result.transfer_strength > 0

    def test_attention_regularization(self):
        """Test attention regularization computation."""
        config = ComposerTransWeaveConfig(
            orthogonality_weight=0.01, sparsity_weight=0.01
        )
        composer_tw = ComposerTransWeave(config)

        # Create diverse attention patterns
        num_heads = 2
        num_cols = 4
        attention = np.zeros((num_heads, num_cols, num_cols))
        attention[0, :, 0] = 1.0  # Head 0 attends to col 0
        attention[1, :, 1] = 1.0  # Head 1 attends to col 1

        reg = composer_tw.compute_attention_regularization(attention)

        assert "orthogonality_loss" in reg
        assert "sparsity_loss" in reg
        assert "total_reg_loss" in reg
        # Diverse patterns should have low orthogonality loss
        assert reg["orthogonality_loss"] < 0.1

    def test_save_load_state(self):
        """Test state serialization."""
        config = ComposerTransWeaveConfig()
        composer_tw = ComposerTransWeave(config)

        # Register a task
        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4
        composer_tw.register_task(
            task_id=0,
            attention_weights=np.random.randn(num_heads, num_cols, num_cols),
            query_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            key_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            value_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
        )

        # Save and load
        state = composer_tw.save_state()
        composer_tw2 = ComposerTransWeave(config)
        composer_tw2.load_state(state)

        assert 0 in composer_tw2.task_representations


class TestShellDemotionTransWeave:
    """Test Within-column Shell Demotion TransWeave."""

    def test_init(self):
        """Test initialization."""
        config = ShellDemotionTransWeaveConfig()
        shell_tw = ShellDemotionTransWeave(config, num_columns=8)
        assert len(shell_tw.column_histories) == 8

    def test_register_shell_state(self):
        """Test shell state registration."""
        config = ShellDemotionTransWeaveConfig(shell_sizes=(4, 8, 4))
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        num_neurons = sum(config.shell_sizes)

        # Create shell assignments
        shell_assignments = np.array([0] * 4 + [1] * 8 + [2] * 4, dtype=np.int32)
        neuron_activities = np.random.rand(num_neurons)

        shell_tw.register_shell_state(
            column_id=0,
            task_id=0,
            shell_assignments=shell_assignments,
            neuron_activities=neuron_activities,
        )

        assert len(shell_tw.column_histories[0]) == 1
        assert shell_tw.column_histories[0][0].task_id == 0

    def test_compute_demotion_no_history(self):
        """Test demotion computation with no history."""
        config = ShellDemotionTransWeaveConfig(shell_sizes=(4, 8, 4))
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        num_neurons = sum(config.shell_sizes)
        result = shell_tw.compute_demotion_transport(
            column_id=0,
            current_activities=np.random.rand(num_neurons),
            current_assignments=np.array([0] * 4 + [1] * 8 + [2] * 4),
        )

        # No history means no candidates
        assert result.diagnostics.get("no_history", False)
        assert len(result.demotion_candidates) == 0

    def test_compute_demotion_with_history(self):
        """Test demotion computation with history."""
        config = ShellDemotionTransWeaveConfig(
            shell_sizes=(4, 8, 4),
            demotion_threshold=0.2,
            max_demotions_per_step=2,
        )
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        num_neurons = sum(config.shell_sizes)
        shell_assignments = np.array([0] * 4 + [1] * 8 + [2] * 4, dtype=np.int32)

        # Register multiple states
        for task_id in range(3):
            activities = np.random.rand(num_neurons)
            # Make inner shell neurons have lower activity over time
            activities[:4] *= 0.5 if task_id > 0 else 1.0
            shell_tw.register_shell_state(
                column_id=0,
                task_id=task_id,
                shell_assignments=shell_assignments,
                neuron_activities=activities,
            )

        # Compute demotion
        result = shell_tw.compute_demotion_transport(
            column_id=0,
            current_activities=np.random.rand(num_neurons) * 0.3,
            current_assignments=shell_assignments,
        )

        assert result.transport_plan.shape == (3, 3)
        assert result.shell_transition_matrix.shape == (3, 3)

    def test_apply_transitions(self):
        """Test applying shell transitions."""
        config = ShellDemotionTransWeaveConfig(shell_sizes=(4, 8, 4))
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        num_neurons = sum(config.shell_sizes)
        current_assignments = np.array([0] * 4 + [1] * 8 + [2] * 4, dtype=np.int32)

        # Create mock demotion result
        from fabricpc.continual.transweave import ShellDemotionResult

        result = ShellDemotionResult(
            demotion_candidates=[
                (0, 0, 1),
                (1, 0, 1),
            ],  # Neurons 0,1 demoted from shell 0 to 1
            promotion_candidates=[(12, 2, 1)],  # Neuron 12 promoted from shell 2 to 1
            transport_plan=np.eye(3),
            shell_transition_matrix=np.eye(3),
            diagnostics={},
        )

        new_assignments, counts = shell_tw.apply_transitions(
            current_assignments, result
        )

        assert counts["demotions_applied"] == 2
        assert counts["promotions_applied"] == 1
        assert new_assignments[0] == 1  # Was 0, now 1
        assert new_assignments[1] == 1
        assert new_assignments[12] == 1  # Was 2, now 1

    def test_get_shell_statistics(self):
        """Test getting shell statistics."""
        config = ShellDemotionTransWeaveConfig(shell_sizes=(4, 8, 4))
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        # No history
        stats = shell_tw.get_shell_statistics(column_id=0)
        assert stats.get("no_history", False)

        # Register state
        num_neurons = sum(config.shell_sizes)
        shell_tw.register_shell_state(
            column_id=0,
            task_id=0,
            shell_assignments=np.array([0] * 4 + [1] * 8 + [2] * 4),
            neuron_activities=np.random.rand(num_neurons),
        )

        stats = shell_tw.get_shell_statistics(column_id=0)
        assert "task_id" in stats
        assert "shell_occupancies" in stats
        assert stats["task_id"] == 0

    def test_save_load_state(self):
        """Test state serialization."""
        config = ShellDemotionTransWeaveConfig(shell_sizes=(4, 8, 4))
        shell_tw = ShellDemotionTransWeave(config, num_columns=4)

        num_neurons = sum(config.shell_sizes)
        shell_tw.register_shell_state(
            column_id=0,
            task_id=0,
            shell_assignments=np.array([0] * 4 + [1] * 8 + [2] * 4),
            neuron_activities=np.random.rand(num_neurons),
        )

        state = shell_tw.save_state()
        shell_tw2 = ShellDemotionTransWeave(config, num_columns=4)
        shell_tw2.load_state(state)

        assert len(shell_tw2.column_histories[0]) == 1


class TestTransWeaveManager:
    """Test unified TransWeave manager."""

    def test_init(self):
        """Test initialization."""
        manager = TransWeaveManager(num_columns=8)
        assert manager.composer_config.enable
        assert manager.shell_config.enable

    def test_init_with_configs(self):
        """Test initialization with custom configs."""
        composer_config = ComposerTransWeaveConfig(enable=False)
        shell_config = ShellDemotionTransWeaveConfig(enable=True)

        manager = TransWeaveManager(
            num_columns=8,
            composer_config=composer_config,
            shell_config=shell_config,
        )

        assert not manager.composer_config.enable
        assert manager.shell_config.enable

    def test_register_task_end(self):
        """Test end-of-task registration."""
        manager = TransWeaveManager(num_columns=4)

        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4
        num_neurons = sum(manager.shell_config.shell_sizes)

        manager.register_task_end(
            task_id=0,
            attention_weights=np.random.randn(num_heads, num_cols, num_cols),
            query_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            key_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            value_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
            column_shell_states={
                0: (
                    np.array([0] * 8 + [1] * 16 + [2] * 8),
                    np.random.rand(num_neurons),
                ),
                1: (
                    np.array([0] * 8 + [1] * 16 + [2] * 8),
                    np.random.rand(num_neurons),
                ),
            },
        )

        assert 0 in manager.composer_transweave.task_representations
        assert len(manager.shell_transweave.column_histories[0]) == 1

    def test_compute_transfers(self):
        """Test computing transfers for new task."""
        manager = TransWeaveManager(num_columns=4)

        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4
        num_neurons = sum(manager.shell_config.shell_sizes)

        # Register 2 tasks
        for task_id in range(2):
            attention = np.random.randn(num_heads, num_cols, num_cols)
            attention = np.abs(attention)
            attention = attention / attention.sum(axis=2, keepdims=True)

            manager.register_task_end(
                task_id=task_id,
                attention_weights=attention,
                query_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                key_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                value_projections=np.random.randn(num_heads, hidden_dim, key_dim),
                output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
                column_shell_states={
                    i: (
                        np.array([0] * 8 + [1] * 16 + [2] * 8),
                        np.random.rand(num_neurons),
                    )
                    for i in range(4)
                },
            )

        # Compute transfers for task 2
        current_attention = np.random.randn(num_heads, num_cols, num_cols)
        current_attention = np.abs(current_attention)
        current_attention = current_attention / current_attention.sum(
            axis=2, keepdims=True
        )

        results = manager.compute_transfers(
            target_task_id=2,
            current_attention=current_attention,
            current_queries=np.random.randn(num_heads, hidden_dim, key_dim),
            current_keys=np.random.randn(num_heads, hidden_dim, key_dim),
            current_values=np.random.randn(num_heads, hidden_dim, key_dim),
            column_current_states={
                i: (
                    np.array([0] * 8 + [1] * 16 + [2] * 8),
                    np.random.rand(num_neurons),
                )
                for i in range(4)
            },
        )

        assert "composer_transfer" in results
        assert "shell_demotions" in results

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        manager = TransWeaveManager(num_columns=4)

        stats = manager.get_summary_stats()

        assert "composer_tasks_registered" in stats
        assert stats["composer_tasks_registered"] == 0

    def test_save_load_state(self):
        """Test state serialization."""
        manager = TransWeaveManager(num_columns=4)

        num_heads, num_cols, hidden_dim, key_dim = 2, 4, 8, 4
        manager.register_task_end(
            task_id=0,
            attention_weights=np.random.randn(num_heads, num_cols, num_cols),
            query_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            key_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            value_projections=np.random.randn(num_heads, hidden_dim, key_dim),
            output_projection=np.random.randn(num_heads * key_dim, hidden_dim),
        )

        state = manager.save_state()
        manager2 = TransWeaveManager(num_columns=4)
        manager2.load_state(state)

        assert 0 in manager2.composer_transweave.task_representations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
