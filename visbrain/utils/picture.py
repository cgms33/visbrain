"""Set of functions for picture managment."""
import numpy as np


def piccrop(im, margin=10):
    """Automatic picture cropping.

    Parameters
    ----------
    im : array_like
        The array of image data. Could be a (N, M) or (N, M, 3/4).
    margin : int | 10
        Number of pixels before and after to condider for cropping
        security.

    Returns
    -------
    imas : array_like
        The cropped figure.
    """
    # ================= Size checking =================
    if im.ndim < 2:
        raise ValueError("im must have at least two dimensions.")
    elif im.ndim == 3:
        imas = im[..., 0:3].mean(axis=2)
    else:
        imas = im

    # ================= Derivative =================
    imdiff_x = np.diff(imas, axis=1)
    imdiff_y = np.diff(imas, axis=0)

    # ================= Cropping start / finish =================
    # x-axis :
    idx_x = np.where(imdiff_x != 0)[1]
    if idx_x.size:
        ncols = imas.shape[1]
        x_min = max(0, idx_x.min() - margin + 1)
        x_max = min(min(ncols, idx_x.max() + margin + 1), ncols)
        sl_x = slice(x_min, x_max)
    else:
        sl_x = slice(None)
    # y-axis :
    idx_y = np.where(imdiff_y)[0]
    if idx_y.size:
        y_min = max(0, idx_y.min() - margin + 1)
        nrows = imas.shape[0]
        y_max = min(min(imas.shape[0], idx_y.max() + margin + 1), nrows)
        sl_y = slice(y_min, y_max)
    else:
        sl_y = slice(None)

    return im[sl_y, sl_x, ...]
