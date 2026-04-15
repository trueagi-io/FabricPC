"""
Reference baseline implementations for continual learning.

Provides standard CL baselines for comparison:
- Naive fine-tuning (no protection against forgetting)
- EWC (Elastic Weight Consolidation)
- Experience replay (buffer-based replay)

These baselines are framework-agnostic and implemented in pure NumPy.
"""

from cl_benchmark.baselines.base import BaselineModel
from cl_benchmark.baselines.naive import NaiveModel
from cl_benchmark.baselines.ewc import EWCModel
from cl_benchmark.baselines.replay import ReplayBuffer

__all__ = [
    "BaselineModel",
    "NaiveModel",
    "EWCModel",
    "ReplayBuffer",
]
