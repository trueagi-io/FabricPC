"""Shared test fixtures and configuration for FabricPC test suite."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax

from fabricpc.core.inference import InferenceSGD

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


def with_inference(structure, inference=None, **kwargs):
    """Return structure with modified inference config for testing."""
    if inference is not None and kwargs:
        raise ValueError("Pass either an inference object or InferenceSGD kwargs")
    new_config = dict(structure.config)
    new_config["inference"] = inference or InferenceSGD(**kwargs)
    return structure._replace(config=new_config)
