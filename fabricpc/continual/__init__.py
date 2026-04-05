"""
FabricPC Continual Learning Module

This module provides tools for continual/sequential learning on Split-MNIST
and similar tasks, ported from the mnist_audit_guided_generality notebook.

Key components:
- SplitMnistLoader: Data loading for sequential digit pair tasks
- SequentialTrainer: Task-by-task training with support selection
- Custom nodes for hierarchical composition architecture
- Support and demotion banks for replay-based continual learning
"""

from fabricpc.continual.config import (
    ExperimentConfig,
    TrainingConfig,
    PatchConfig,
    ColumnConfig,
    ShellConfig,
    SupportConfig,
    TypingConfig,
    ComposerConfig,
    HierarchyConfig,
    CloudConfig,
    TransWeaveConfig,
    CheckpointConfig,
    AuditConfig,
    PredictorConfig,
    make_config,
)

from fabricpc.continual.data import (
    SplitMnistLoader,
    TaskData,
    build_split_mnist_loaders,
)

from fabricpc.continual.trainer import SequentialTrainer, TaskRunSummary

__all__ = [
    # Config
    "ExperimentConfig",
    "TrainingConfig",
    "PatchConfig",
    "ColumnConfig",
    "ShellConfig",
    "SupportConfig",
    "TypingConfig",
    "ComposerConfig",
    "HierarchyConfig",
    "CloudConfig",
    "TransWeaveConfig",
    "CheckpointConfig",
    "AuditConfig",
    "PredictorConfig",
    "make_config",
    # Data
    "SplitMnistLoader",
    "TaskData",
    "build_split_mnist_loaders",
    # Training
    "SequentialTrainer",
    "TaskRunSummary",
]
