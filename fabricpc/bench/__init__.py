"""fabricpc.bench: a benchmark suite where every result comes from one command.

Run ``python -m fabricpc.bench list`` to see the rows, or
``python -m fabricpc.bench <row-id>`` to run one.
"""

from fabricpc.bench.definitions.registry import ALGORITHMS, ROWS, BenchmarkRow

__all__ = ["ALGORITHMS", "ROWS", "BenchmarkRow"]
