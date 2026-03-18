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


def with_inference(structure, **kwargs):
    """Return structure with modified inference config for testing."""
    new_config = dict(structure.config)
    new_config["inference"] = InferenceSGD(**kwargs)
    return structure._replace(config=new_config)
