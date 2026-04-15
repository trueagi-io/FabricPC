"""
Evaluation module for continual learning benchmarks.

Provides the main entry points for running benchmarks:
- BenchmarkConfig: Configuration for benchmark runs
- BenchmarkRunner: Executes benchmarks and computes metrics
- BenchmarkResults: Container for benchmark results
"""

from cl_benchmark.evaluation.protocol import BenchmarkConfig
from cl_benchmark.evaluation.results import BenchmarkResults
from cl_benchmark.evaluation.runner import BenchmarkRunner

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResults",
    "BenchmarkRunner",
]
