"""The model zoo: trained weights from benchmark trials, saved to disk.

Layout, one folder per trial plus a small JSON sidecar:

    zoo/<row id>/trial<i>/        the Orbax checkpoint of the params
    zoo/<row id>/trial<i>.json    row id, trial, seed, and the score

Weights are saved with Orbax's StandardCheckpointer, so they load back
bit-for-bit. Loading needs a template of the same shape, which the row's
model factory provides. When the library's own checkpoint module lands,
this file becomes a thin wrapper around it.
"""

import json
from pathlib import Path
from typing import Tuple

import jax
import orbax.checkpoint as ocp

from fabricpc.bench.definitions.registry import BenchmarkRow


def checkpoint_dir(zoo_dir, row_id: str, trial: int) -> Path:
    return Path(zoo_dir) / row_id / f"trial{trial}"


def save_params(zoo_dir, row: BenchmarkRow, trial: int, params, *, meta: dict) -> Path:
    """Save trained ``params`` for one trial and return the checkpoint path."""
    path = checkpoint_dir(zoo_dir, row.id, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(path.resolve(), params, force=True)
    ckptr.wait_until_finished()
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({"row_id": row.id, "trial": trial, **meta}, indent=2))
    return path


def load_params(zoo_dir, row: BenchmarkRow, trial: int) -> Tuple[object, object]:
    """Load a trial's params. Returns ``(params, structure)`` ready to evaluate.

    The structure is rebuilt from the row (it is code, not data); the key
    used to build the template does not matter because every array is
    overwritten by the checkpoint.
    """
    template, structure = row.model_factory(jax.random.PRNGKey(0))
    path = checkpoint_dir(zoo_dir, row.id, trial)
    params = ocp.StandardCheckpointer().restore(path.resolve(), template)
    return params, structure
