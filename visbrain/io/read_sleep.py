"""Load sleep files.

This file contain functions to load PSG and hypnogram files.
"""
import os
import io
import numpy as np
import datetime
from warnings import warn
from scipy.stats import iqr
from mne.filter import resample
import logging

from .rw_utils import get_file_ext
from .rw_hypno import (read_hypno, oversample_hypno)
from .dialog import dialog_load
from .dependencies import is_mne_installed
from ..utils import vispy_array
from ..io import merge_annotations
from ..config import PROFILER

logger = logging.getLogger('visbrain')

__all__ = ['ReadSleepData', 'read_elan', 'mne_switch']


class ReadSleepData(object):
    """Main class for reading sleep data."""

    def __init__(self, data, channels, sf, hypno, href,
                 downsample, kwargs_mne, annotations):
        """Init."""
        # ========================== LOAD DATA ==========================
        # Dialog window if data is None :
        if data is None:
            data = dialog_load(self, "Open dataset", '',
                               "BrainVision (*.vhdr);;EDF (*.edf);;"
                               "GDF (*.gdf);;BDF (*.bdf);;Elan (*.eeg);;"
                               "EGI (*.egi);;MFF (*.mff);;CNT (*.cnt);;"
                               "EEGLab (*.set);;REC (*.rec);;Eximia (*.nxe)")
            upath = os.path.split(data)[0]
        else:
            upath = ''

        if isinstance(data, str):  # file is defined
            # ---------- USE SLEEP or MNE ----------
            # Find file extension :
            file, ext = get_file_ext(data)
            # Get if the file has to be loaded using Sleep or MNE python :
            use_mne = True if ext != '.eeg' else False

            if use_mne:
                is_mne_installed(raise_error=True)

            # ---------- LOAD THE FILE ----------
            if use_mne:  # Load using MNE functions
                logger.debug("Loading file using MNE-python")
                kwargs_mne['preload'] = True
                args = mne_switch(file, ext, downsample, **kwargs_mne)
            else:  # Load using Sleep functions
                logger.debug("Loading file using Sleep")
                args = read_elan(file + ext, downsample)
            # Get output arguments :
            (sf, downsample, data, channels, n, offset, annot) = args
            info = ("Data successfully loaded (%s):"
                    "\n- Sampling-frequency : %.2fHz"
                    "\n- Number of time points (before down-sampling): %i"
                    "\n- Down-sampling frequency : %.2fHz"
                    "\n- Number of time points (after down-sampling): %i"
                    "\n- Number of channels : %i"
                    )
            n_channels, n_pts_after = data.shape
            logger.info(info % (file + ext, sf, n, downsample, n_pts_after,
                                n_channels))
            PROFILER("Data file loaded", level=1)

        elif isinstance(data, np.ndarray):  # array of data is defined
            if not isinstance(sf, (int, float)):
                raise ValueError("When passing raw data, the sampling "
                                 "frequency parameter, sf, must either be an "
                                 "integer or a float.")
            file = annot = None
            offset = datetime.time(0, 0, 0)
            n = data.shape[1]
            dsf = downsample / sf if downsample is not None else 1
            data = resample(data, dsf)
        else:
            raise IOError("The data should either be a string which refer to "
                          "the path of a file or an array of raw data of shape"
                          " (n_electrodes, n_time_points).")

        # Keep variables :
        self._file = file
        self._annot_file = np.c_[merge_annotations(annotations, annot)]
        self._N = n
        self._N_ds = data.shape[1]
        self._sfori = float(sf)
        self._toffset = offset.hour * 3600. + offset.minute * 60. + \
            offset.second
        self._sf = float(downsample) if downsample is not None else float(sf)
        time = np.arange(self._N_ds) / self._sf

        # ========================== LOAD HYPNOGRAM ==========================
        # Dialog window for hypnogram :
        if hypno is None:
            hypno = dialog_load(self, "Open hypnogram", upath,
                                "Text file (*.txt);;Elan (*.hyp);;"
                                "CSV file (*.csv);;EDF+ file(*.edf);"
                                ";All files (*.*)")
            hypno = None if hypno == '' else hypno
        if isinstance(hypno, str):  # (*.hyp / *.txt / *.csv)
            hypno, _ = read_hypno(hypno, time=time, datafile=file)
            # Oversample hypno
            hypno = oversample_hypno(hypno, self._N_ds)
            PROFILER("Hypnogram file loaded", level=1)

        # ========================== CHECKING ==========================
        # ---------- DATA ----------
        # Check data shape :
        if data.ndim is not 2:
            raise ValueError("The data must be a 2D array")
        nchan, npts = data.shape

        # ---------- CHANNELS ----------
        if (channels is None) or (len(channels) != nchan):
            warn("The number of channels must be " + str(nchan) + ". Default "
                 "channel names will be used instead.")
            channels = ['chan' + str(k) for k in range(nchan)]
        # Clean channel names :
        patterns = ['eeg', 'EEG', 'ref']
        chanc = []
        for c in channels:
            # Remove informations after . :
            c = c.split('.')[0]
            c = c.split('-')[0]
            # Exclude patterns :
            for i in patterns:
                c = c.replace(i, '')
            # Remove space :
            c = c.replace(' ', '')
            c = c.strip()
            chanc.append(c)

        # ---------- STAGE ORDER ----------
        # href checking :
        absref = ['art', 'wake', 'n1', 'n2', 'n3', 'rem']
        absint = [-1, 0, 1, 2, 3, 4]
        if href is None:
            href = absref
        elif (href is not None) and isinstance(href, list):
            # Force lower case :
            href = [k.lower() for k in href]
            # Check that all stage are present :
            for k in absref:
                if k not in href:
                    raise ValueError(k + " not found in href.")
            # Force capitalize :
            href = [k.capitalize() for k in href]
            href[href.index('Rem')] = 'REM'
        else:
            raise ValueError("The href parameter must be a list of string and"
                             " must contain 'art', 'wake', 'n1', 'n2', 'n3' "
                             "and 'rem'")
        # Conversion variable :
        absref = ['Art', 'Wake', 'N1', 'N2', 'N3', 'REM']
        conv = {absint[absref.index(k)]: absint[i] for i, k in enumerate(href)}

        # ---------- HYPNOGRAM ----------
        if hypno is None:
            hypno = np.zeros((npts,), dtype=np.float32)
        else:
            n = len(hypno)
            # Check hypno values :
            if (hypno.min() < -1.) or (hypno.max() > 4) or (n != npts):
                warn("\nHypnogram values must be comprised between -1 and 4 "
                     "(see Iber et al. 2007). Use:\n-1 -> Art (optional)\n 0 "
                     "-> Wake\n 1 -> N1\n 2 -> N2\n 3 -> N4\n 4 -> REM\nEmpty "
                     "hypnogram will be used instead")
                hypno = np.zeros((npts,), dtype=np.float32)

        # ---------- SCALING ----------
        # Check amplitude of the data and if necessary apply re-scaling
        # Assume that the inter-quartile amplitude of EEG data is ~50 uV
        iqr_data = iqr(data)
        if iqr_data < 1:
            mult_fact = np.floor(np.log10(50 / iqr_data))
            warn("Wrong data amplitude. Multiplying data amplitude by 10^%i"
                 % mult_fact)
            data *= 10 ** mult_fact
        # ---------- CONVERSION ----------=
        # Convert data and hypno to be contiguous and float 32 (for vispy):
        self._data = vispy_array(data)
        self._hypno = vispy_array(hypno)
        self._time = vispy_array(time)
        self._channels = chanc
        self._href = href
        self._hconv = conv
        PROFILER("Check data", level=1)

###############################################################################
###############################################################################
#                               LOAD FILES
###############################################################################
###############################################################################

def read_elan(path, downsample):
    """Read data from a ELAN (eeg) file.

    Elan format specs: http: // elan.lyon.inserm.fr/

    Parameters
    ----------
    path : str
        Filename(with full path) to Elan .eeg file
    downsample : int
        Down-sampling frequency.

    Returns
    -------
    sf : int
        The original sampling frequency.
    downsample :
        The downsampling frequency
    data : array_like
        The downsampled data organised as (n_channels, n_points)
    chan : list
        The list of channel's names.
    n : int
        Number of samples before down-sampling.
    start_time : array_like
        Starting time of the recording (hh:mm:ss)
    annotations : array_like
        Array of annotations.
    """
    header = path + '.ent'

    assert os.path.isfile(path)
    assert os.path.isfile(header)

    # Read .ent file
    ent = np.genfromtxt(header, delimiter='\n', usecols=[0],
                        dtype=None, skip_header=0, encoding='utf-8')

    # eeg file version
    eeg_version = ent[0]

    if eeg_version == 'V2':
        nb_oct = 2
        formread = '>i2'
    elif eeg_version == 'V3':
        nb_oct = 4
        formread = '>i4'

    # Sampling rate
    sf = 1. / float(ent[8])

    # Record starting time
    if ent[4] != "No time":
        hour, minutes, sec = ent[4].split(':')
        start_time = datetime.time(int(hour), int(minutes), int(sec))
        day, month, year = ent[3].split(':')
    else:
        start_time = datetime.time(0, 0, 0)

    # Channels
    nb_chan = np.int(ent[9])
    nb_chan = nb_chan

    # Last 2 channels do not contain data
    nb_chan_data = nb_chan - 2
    chan_list = slice(nb_chan_data)
    chan = ent[10:10 + nb_chan_data]

    # Gain
    gain = np.zeros(nb_chan)
    offset1 = 9 + 3 * nb_chan
    offset2 = 9 + 4 * nb_chan
    offset3 = 9 + 5 * nb_chan
    offset4 = 9 + 6 * nb_chan

    for i in np.arange(1, nb_chan + 1):

        min_an = float(ent[offset1 + i])
        max_an = float(ent[offset2 + i])
        min_num = float(ent[offset3 + i])
        max_num = float(ent[offset4 + i])

        gain[i - 1] = (max_an - min_an) / (max_num - min_num)
    if gain.dtype != np.float32:
        gain = gain.astype(np.float32, copy=False)

    # Load memmap
    nb_bytes = os.path.getsize(path)
    nb_samples = int(nb_bytes / (nb_oct * nb_chan))

    m_raw = np.memmap(path, dtype=formread, mode='r',
                      shape=(nb_chan, nb_samples), order={'F'})

    # Get original signal length :
    n = m_raw.shape[1]
    chan = list(chan)

    # Downsampling
    dsf = sf / downsample if downsample is not None else 1.
    if float(dsf).is_integer():
        # Decimate and multiply by gain :
        data = m_raw[chan_list, ::int(dsf)] * gain[chan_list][..., np.newaxis]
    else:
        # Load in full and then downsample (slow for large files)
        data = m_raw[chan_list, :] * gain[chan_list][..., np.newaxis]
        data = resample(data, 1 / dsf)

    return sf, downsample, data, chan, n, start_time, None


def mne_switch(file, ext, downsample, preload=True, **kwargs):
    """Read sleep datasets using mne.io.

    Parameters
    ----------
    file : string
        Filename (without extension).
    ext : string
        File extension (e.g. '.edf'').
    preload : bool | True
        Preload data in memory.
    kwargs : dict | {}
        Further arguments to pass to the mne.io.read function.

    Returns
    -------
    sf : float
        The original sampling-frequency.
    downsample : float
        The down-sampling frequency used.
    data : array_like
        The downsampled data of shape (n_channels, n_points)
    channels : list
        List of channel names.
    n : int
        Number of time points before down-sampling.
    start_time : datetime.time
        The time offset.
    """
    from mne import io

    # Get full path :
    path = file + ext

    # Preload :
    if preload is False:
        preload = 'temp.dat'
    kwargs['preload'] = preload

    if ext.lower() in ['.edf', '.bdf', '.gdf']:  # EDF / BDF / GDF
        raw = io.read_raw_edf(path, **kwargs)
    elif ext.lower == '.set':   # EEGLAB
        raw = io.read_raw_eeglab(path, **kwargs)
    elif ext.lower() in ['.egi', '.mff']:  # EGI / MFF
        raw = io.read_raw_egi(path, **kwargs)
    elif ext.lower() == '.cnt':  # CNT
        raw = io.read_raw_cnt(path, **kwargs)
    elif ext.lower() == '.vhdr':  # BrainVision
        raw = io.read_raw_brainvision(path, **kwargs)
    elif ext.lower() == '.nxe': # Eximia
        raw = io.read_raw_eximia(path, **kwargs)
    else:
        raise IOError("File not supported by mne-python.")

    raw.pick_types(meg=True, eeg=True, ecg=True, emg=True)  # Remove stim lines
    sf = raw.info['sfreq']
    n = raw._data.shape[1]

    # Downsample
    if downsample is not None:
        dsf = sf / downsample if downsample is not None else 1.
        if float(dsf).is_integer():
            # Decimate
            data = raw._data[:, ::int(dsf)]
        else:
            # Use MNE built-in function
            raw.resample(downsample, npad='auto')
            data = raw._data
    else:
        data = raw._data

    channels = raw.info['ch_names']

    # Conversion Volt (MNE) to microVolt (Visbrain):
    if raw._raw_extras[0] is not None and 'units' in raw._raw_extras[0]:
        units = raw._raw_extras[0]['units'][0:data.shape[0]]
        data /= np.array(units).reshape(-1, 1)

    start_time = datetime.time(0, 0, 0)  # raw.info['meas_date']
    anot = raw.annotations

    return sf, downsample, data, channels, n, start_time, anot
