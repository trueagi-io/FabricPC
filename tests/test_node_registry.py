"""Legacy node registry tests.

Registry functionality was removed in favor of direct node-class construction.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Node registry removed; tests retained as placeholder until rewritten for object API"
)


def test_registry_removed_placeholder():
    assert True
