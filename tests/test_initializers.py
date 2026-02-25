"""Legacy initializer registry tests.

Registry APIs were removed in favor of direct initializer-object construction.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Initializer registry removed; tests pending rewrite for direct object API"
)


def test_initializer_registry_removed_placeholder():
    assert True
