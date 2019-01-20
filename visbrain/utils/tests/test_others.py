"""Test functions in others.py."""
from visbrain.utils.others import set_if_not_none


class TestOthers(object):
    """Test functions in others.py."""

    def test_set_if_not_none(self):
        """Test function set_if_not_none."""
        a = 5.
        assert set_if_not_none(a, None) == 5.
        assert set_if_not_none(a, 10., False) == 5.
        assert set_if_not_none(a, 10.) == 10.
