"""Test functions in physio.py."""
import numpy as np

from visbrain.utils.picture import piccrop


class TestPicture(object):
    """Test functions in picture.py."""

    def _compare_shapes(self, im, shapes):
        im_shapes = [k.shape for k in im]
        assert np.array_equal(im_shapes, shapes)

    def test_piccrop(self):
        """Test function piccrop."""
        pic = np.array([[0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 1., 1., 1., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]])
        destination = np.array([[0., 1., 0.],
                                [1., 1., 1.],
                                [0., 1., 0.]])
        assert np.array_equal(piccrop(pic, margin=0), destination)
