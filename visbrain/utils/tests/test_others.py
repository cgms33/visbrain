"""Test functions in others.py."""
from visbrain.utils.others import set_if_not_none
from visbrain.io.path import get_data_path


class TestOthers(object):
    """Test functions in others.py."""

    def test_set_if_not_none(self):
        """Test function set_if_not_none."""
        a = 5.
        assert set_if_not_none(a, None) == 5.
        assert set_if_not_none(a, 10., False) == 5.
        assert set_if_not_none(a, 10.) == 10.

    def test_get_data_path(self):
        """Test function get_data_path."""
        assert isinstance(get_data_path(), str)
