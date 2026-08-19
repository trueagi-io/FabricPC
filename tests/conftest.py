"""Shared test fixtures and configuration for FabricPC test suite."""

import os

# Settings outside setup_jax's scope; everything it covers is set through it below.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax

from fabricpc import setup_jax
from fabricpc.core.inference import InferenceSGD

setup_jax("cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


def with_inference(structure, **kwargs):
    """Return structure with modified inference config for testing."""
    new_config = dict(structure.config)
    new_config["inference"] = InferenceSGD(**kwargs)
    return structure._replace(config=new_config)
