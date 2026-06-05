"""
This module provides functions for analysing stimulation over time.

The functions detect stimulation onset and offset times from time-series
signals and compute stimulation durations from simulation or experimental CSV
files.

Functions
---------
get_stim_timing(time, signal)
    Detect stimulation pulse trains in a signal and return their onset and
    offset time.

get_stim_dur(filepaths)
    Compute mean stimulation duration(s) from one or multiple CSV files
    containing time-series stimulation data.
"""

import numpy as np
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_stim_timing(time, signal):
    """
    Detect stimulation onset and offset time in a signal.

    This function identifies the onset and offset times of pulse trains in a
    time-series signal by thresholding and grouping consecutive pulses into
    trains based on temporal gaps.

    Parameters
    ----------
    time : array-like
        1D array of time values corresponding to `signal`.
    signal : array-like
        1D array of signal values representing stimulation over time.

    Returns
    -------
    t_stim_on : numpy.ndarray
        Array of onset times for each detected pulse train.
    t_stim_off : numpy.ndarray
        Array of offset times for each detected pulse train.

    Notes
    -----
    - A threshold is computed as the midpoint between the minimum and maximum
      of the signal to binarize it.
    - Pulse starts and stops are detected using differences in the binarized
      signal.
    - Pulse trains are defined by grouping pulses separated by gaps smaller
      than a characteristic pulse width.
    - If unusually large pulse widths are detected (e.g., >100 samples),
      the grouping threshold is reset.
    - The helper function `createBlockSignal` is used to construct the output
      block signal.

    Examples
    --------
    >>> t_on, t_off = get_stim_timing(time, signal)

    >>> t_on
    array([0.5, 2.0])

    >>> t_off
    array([1.0, 2.5])
    """
    # Determine threshold as midpoint between min and max
    threshold = (np.max(signal) + np.min(signal)) / 2

    # Binarize signal
    binary_signal = np.where(signal > threshold, 1, 0)

    # Find pulse start and stop indices
    i_start_pulse = np.where(np.diff(binary_signal, prepend=0) == 1)[0]
    i_stop_pulse = np.where(np.diff(binary_signal, prepend=0) == -1)[0]

    if len(i_start_pulse) == 0 or len(i_stop_pulse) == 0:
        return [], [], []

    # Estimate characteristic pulse width
    i_start_pulse_same = i_start_pulse
    if i_start_pulse_same[0] == 0:
        i_start_pulse_same = i_start_pulse_same[1:]

    pulse_widths = i_stop_pulse - i_start_pulse_same
    pulse_width_median = np.median(pulse_widths)

    # Handle unusually large pulse widths
    if any(abs(pulse_widths) > 100):
        pulse_width_median = 0

    # Group pulses into trains
    i_start_train = [int(i_start_pulse[0])]
    i_stop_train = []

    for i in range(1, len(i_start_pulse)):
        if i_start_pulse[i] - i_stop_pulse[i - 1] > pulse_width_median:
            i_stop_train.append(int(i_stop_pulse[i - 1]))
            i_start_train.append(int(i_start_pulse[i]))

    # Add final stop index
    i_stop_train.append(int(i_stop_pulse[-1]))

    # Output
    t_stim_on = time[i_start_train]
    t_stim_off = time[i_stop_train]

    return t_stim_on, t_stim_off

def get_stim_dur(filepaths):
    """
    Compute mean stimulus duration(s) from one or more CSV files.

    Each file is expected to contain time-series data where stimulation onset
    and offset can be detected using `get_stim_timing`. The function extracts
    stimulus durations and returns their mean for each file. If a file cannot
    be processed, NaN is returned for that entry.

    Parameters
    ----------
    filepaths : str or list of str
        Path or list of paths to CSV file(s) containing the data.

    Returns
    -------
    stim_durations : numpy.ndarray
        Array of mean stimulus durations for each input file. If a file
        cannot be processed, the corresponding value is `np.nan`.

    Notes
    -----
    - The CSV file is read using `pandas.read_csv`, transposed, and converted
      to a NumPy array.
    - The function assumes the data contains at least three rows corresponding
      to time and stimulus signal.
    - The helper function `get_stim_timing` must return stimulation onset and
      offset times.

    Examples
    --------
    >>> get_stim_dur("data.csv")
    array([0.52])

    >>> get_stim_dur(["file1.csv", "file2.csv"])
    array([0.52, nan])
    """
    
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    stim_durations = []

    for filepath in filepaths:
        try:
            data = pd.read_csv(filepath).T.to_numpy()
            time, _, stim, *_ = data

            t_stim_on, t_stim_off = get_stim_timing(
                time[0:-1], stim[0:-1]
            )

            stim_dur = t_stim_off - t_stim_on
            stim_durations.append(np.mean(stim_dur))

        except Exception:
            stim_durations.append(np.nan)

    return np.array(stim_durations)
        
