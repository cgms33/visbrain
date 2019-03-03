"""Test functions in mesh.py."""
import numpy as np

from visbrain.utils.mesh import vispy_array


class TestMesh(object):
    """Test functions in mesh.py."""

    def test_vispy_array(self):
        """Test vispy_array function."""
        mat = np.random.randint(0, 10, (10, 30))
        mat_convert = vispy_array(mat, np.float64)
        assert mat_convert.flags['C_CONTIGUOUS']
        assert mat_convert.dtype == np.float64
