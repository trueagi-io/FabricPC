"""Search space specifications for glucose Optuna tuning.

Loads parameter search spaces from YAML config files under
``examples/configs/``, with a generic ``suggest_from_spec`` helper that
maps them onto an Optuna trial.  Both the transformer and Hopfield
tuners import from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import yaml

ParamSpec = dict[str, Any]
SearchSpace = dict[str, ParamSpec]

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_space(path: str | Path) -> SearchSpace:
    """Load a search space from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


TRANSFORMER_SPACE: SearchSpace = load_space(
    _CONFIGS_DIR / "glucose_transformer_space.yaml"
)
HOPFIELD_SPACE: SearchSpace = load_space(
    _CONFIGS_DIR / "glucose_hopfield_space.yaml"
)


def suggest_from_spec(
    trial: optuna.Trial,
    space: SearchSpace,
) -> dict[str, Any]:
    """Sample all parameters from a search space spec using the Optuna trial."""
    result: dict[str, Any] = {}

    deferred: list[tuple[str, ParamSpec]] = []
    for name, spec in space.items():
        if "condition" in spec:
            deferred.append((name, spec))
        else:
            result[name] = _suggest_one(trial, name, spec)

    for name, spec in deferred:
        cond = spec["condition"]
        if result.get(cond["parent"]) == cond["value"]:
            result[name] = _suggest_one(trial, name, spec)
        else:
            result[name] = spec["default"]

    return result


def _suggest_one(trial: optuna.Trial, name: str, spec: ParamSpec) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["values"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if kind == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    raise ValueError(f"Unknown param type {kind!r} for {name!r}")
