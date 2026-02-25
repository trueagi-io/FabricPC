"""Legacy energy registry tests.

Registry APIs were removed in favor of direct energy-object construction.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Energy registry removed; tests pending rewrite for direct object API"
)


def test_energy_registry_removed_placeholder():
    assert True
