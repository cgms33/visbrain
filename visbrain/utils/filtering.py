"""Set of tools to filter data."""

import numpy as np
from mne.filter import filter_data
from mne.time_frequency import morlet as _morlet_wlt
from scipy.signal import detrend

__all__ = ('filt', 'morlet', 'ndmorlet', 'morlet_power', 'PrepareData')

#############################################################################
# FILTERING
#############################################################################


def filt(sf, f, x, btype='bandpass'):
    """Filt data.

    Parameters
    ----------
    sf : float
        The sampling frequency
    f : array_like
        Frequency vector (2,)
    x : array_like
        The data to filt.
    btype : {'bandpass', 'bandstop', 'highpass', 'lowpass'}
        If highpass, the first value of f will be used. If lowpass
        the second value of f will be used.

    Returns
    -------
    xfilt : array_like
        Filtered data.
    """
    if btype == 'bandpass':
        low, high = f[0], f[1]
    elif btype == 'bandstop':
        high, low = f[0], f[1]
    elif btype == 'highpass':
        low = f[0]
        high = None
    elif btype == 'lowpass':
        low = None
        high = f[1]

    return filter_data(x.astype(np.float64), sf, low, high, method='fir',
                       phase='zero', fir_window='hamming', fir_design='firwin',
                       pad='reflect_limited', verbose=0)

#############################################################################
# WAVELET
#############################################################################


def morlet(x, sf, f, width=7):
    """Complex decomposition of a signal x using the morlet wavelet.

    Parameters
    ----------
    x : array_like
        The signal to use for the complex decomposition. Must be
        a vector of length N.
    sf : float
        Sampling frequency
    f : float
        Central frequency
    width : float
        Number of oscillations of the wavelet

    Returns
    -------
    xout: array_like
        The complex decomposition of the signal x.
    """
    # Get the wavelet and convolve with signal
    m = _morlet_wlt(sf, freqs=[f], n_cycles=width)[0]
    return np.convolve(x, m, mode='same')


def ndmorlet(x, sf, f, axis=0, get=None, width=7.0):
    """Complex decomposition using Morlet's wlt for a multi-dimentional array.

    Parameters
    ----------
    x : array_like
        The signal to use for the complex decomposition.
    sf : float
        Sampling frequency
    f : array_like
        Frequency vector of shape (2,)
    axis : integer | 0
        Specify the axis where is located the time dimension
    get : {'amplitude', 'phase', 'power'}
        Specify whether to return the amplitude, phase or power of the
        analytic signal.
    width : float | 7.0
        Width of the wavelet

    Returns
    -------
    xf : array, same shape as x
        Complex decomposition of x.
    """
    # Get the wavelet :
    m = _morlet_wlt(sf, freqs=[f], n_cycles=width)[0]

    # Define a morlet function :
    def morlet_fcn(xt):
        return np.convolve(xt, m, mode='same')

    xf = np.apply_along_axis(morlet_fcn, axis, x)

    # Get amplitude / power / phase :
    if get == 'amplitude':
        return np.abs(xf)
    elif get == 'power':
        return np.square(np.abs(xf))
    elif get == 'phase':
        return np.angle(xf)


def morlet_power(x, freqs, sf, norm=True):
    """Compute bandwise-normalized power of data using morlet wavelet.

    Parameters
    ----------
    x : array_like
        Row vector signal.
    freqs : array_like
        Frequency bands for power computation. The power will be computed
        using successive frequency band (e.g freqs=(1., 2, .3)).
    sf : float
        Sampling frequency.
    norm : bool | True
        If True, return bandwise normalized band power
        (For each time point, the sum of power in the 4 band equals 1)

    Returns
    -------
    xpow : array_like
        The power in the specified frequency bands of shape
        (len(freqs)-1, npts).
    """
    # Build frequency vector :
    f = np.c_[freqs[0:-1], freqs[1::]].mean(1)
    # Get wavelet transform :
    xpow = np.zeros((len(f), len(x)), dtype=np.float)
    for num, k in enumerate(f):
        xpow[num, :] = np.abs(morlet(x, sf, k))
    # Compute inplace power :
    np.power(xpow, 2, out=xpow)
    # Normalize by the band sum :
    if norm:
        sum_pow = xpow.sum(0).reshape(1, -1)
        np.divide(xpow, sum_pow, out=xpow)
    return xpow


class PrepareData(object):
    """Prepare data before plotting.

    This class group a set of signal processing tools including :
        - De-meaning
        - De-trending
        - Filtering
        - Decomposition (filter / amplitude / power / phase)
    """

    def __init__(self, axis=0, demean=False, detrend=False, filt=False,
                 fstart=12., fend=16., btype='bandpass', dispas='filter'):
        """Init."""
        # Axis along which to perform preparation :
        self.axis = axis
        # Demean and detrend :
        self.demean = demean
        self.detrend = detrend
        # Filtering :
        self.filt = filt
        self.fstart, self.fend = fstart, fend
        self.btype = btype
        self.dispas = dispas

    def __bool__(self):
        """Return if data have to be prepared."""
        return any([self.demean, self.detrend, self.filt])

    def _prepare_data(self, sf, data, time):
        """Prepare data before plotting."""
        # ============= DEMEAN =============
        if self.demean:
            mean = np.mean(data, axis=self.axis, keepdims=True)
            np.subtract(data, mean, out=data)

        # ============= DETREND =============
        if self.detrend:
            data = detrend(data, axis=self.axis)

        # ============= FILTERING =============
        if self.filt:
            if self.dispas == 'filter':
                data = filt(sf, np.array([self.fstart, self.fend]), data,
                            btype=self.btype)
            else:
                # Compute ndwavelet :
                f = np.array([self.fstart, self.fend]).mean()
                data = ndmorlet(data, sf, f, axis=self.axis, get=self.dispas)

        return data

    def update(self):
        """Update object."""
        if self._fcn is not None:
            self._fcn()
