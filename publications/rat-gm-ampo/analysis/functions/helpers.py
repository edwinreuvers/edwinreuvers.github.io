"""
This module provides helper functions for loading simulation results and
computing average mechanical power output (AMPO).

Functions
---------
load_sims(cf_set, fts_set, mle_set, mus, data_dir)
    Load simulation files for a grid of SSC parameters and compute AMPO.

get_ampo(filepaths)
    Compute AMPO from one or multiple simulation CSV files.
"""

import os
import numpy as np
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_sims(cf_set, fts_set, mle_set, mus, data_dir):
    """
    Load simulation data and compute AMPO values for given SSC parameter grid.

    This function iterates over combinations of cycle frequency, FTS and
    MTC length excursion, loads the corresponding CSV files, computes AMPO and
    returns an array with one value for each parameter combination.

    Parameters
    ----------
    cf_set : array-like
        Cycle frequency values [Hz].
    fts_set : array-like
        Fraction of cycle time spent shortening [-].
    mle_set : array-like
        MTC length excursion values [m].
    mus : str
        Base filename prefix used for constructing file names.
    data_dir : str
        Directory path containing the CSV simulation files.

    Returns
    -------
    np.ndarray
        Array of computed AMPO values with shape:
        (len(cf_set), len(fts_set), len(mle_set)),
        or a squeezed version if dimensions are singleton.

    Notes
    -----
    If a file cannot be read or processed, the corresponding entry
    is set to NaN.

    Examples
    --------
    >>> load_sims([2.0], [0.5, 0.75], [0.004], "GMe1", "./data/")
    array([...])
    """
    
    cf_set = np.atleast_1d(cf_set)
    fts_set = np.atleast_1d(fts_set)
    mle_set = np.atleast_1d(mle_set)

    ampo_set = np.full(
        (len(cf_set), len(fts_set), len(mle_set)),
        np.nan
    )

    for i_cf, cf in enumerate(cf_set):
        for i_fts, fts in enumerate(fts_set):
            for i_mle, mle in enumerate(mle_set):
                try:
                    file_name = (
                        f"{mus}_cf{cf:0.1f}Hz_fts{fts:0.2f}_mle{mle*1e3:0.1f}mm"
                    )
                    filepath = os.path.join(data_dir, file_name + ".csv")
                    df = pd.read_csv(filepath)
                    data = df.to_numpy()

                    time, lmtc, _, fsee = data.T[:4]

                    w_mech = -np.trapezoid(fsee, lmtc)
                    ampo_set[i_cf, i_fts, i_mle] = w_mech * cf

                except Exception:
                    ampo_set[i_cf, i_fts, i_mle] = np.nan

    return np.squeeze(ampo_set)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_ampo(filepaths):
    """
    Compute AMPO from one or more CSV files.

    This function loads time-series data from CSV file(s) and computes AMPO as:

        AMPO = -∫ fsee d(lmtc) / time_end

    The function supports flexible input structures, including a single file,
    1D, 2D, or 3D arrays of file paths. Each file is expected to contain at
    least the following rows after transposition:
    time, lmtc, stim, fsee.

    Parameters
    ----------
    filepaths : str or array-like of str
        Path(s) to CSV file(s). Can be:
        - str: single file path
        - 1D list/array: multiple files
        - 2D list/array: grid of file paths
        - 3D list/array: 3D block of file paths

    Returns
    -------
    float or numpy.ndarray
        Computed AMPO values:
        - float for a single file
        - 1D array for 1D input
        - 2D array for 2D input
        - 3D array for 3D input
        Entries are np.nan when computation fails for a file.

    Notes
    -----
    - CSV files are read using pandas and transposed (`.T.to_numpy()`).
    - The function assumes consistent row ordering across files.
    - Integration is performed using the trapezoidal rule.
    - Division by `time[-1]` normalizes the metric by total duration.
    - Any file that cannot be read or processed is assigned np.nan.

    Raises
    ------
    ValueError
        If the input structure has more than 3 dimensions or is unsupported.

    Examples
    --------
    >>> get_ampo("file.csv")
    0.42

    >>> get_ampo(["f1.csv", "f2.csv"])
    array([0.42, nan])

    >>> get_ampo([[ "a.csv", "b.csv" ],
    ...           [ "c.csv", "d.csv" ]])
    array([[0.41, 0.38],
           [0.44, nan]])
    """

    # Case 0: Single string input
    if isinstance(filepaths, str):
        try:
            data = pd.read_csv(filepaths).T.to_numpy()
            time, lmtc, stim, fsee, *_ = data
            return -np.trapz(fsee, lmtc) / time[-1]
        except Exception:
            return np.nan

    # Convert to NumPy array to check dimensions
    filepaths_arr = np.array(filepaths, dtype=object)

    # Case 12: 1D list of filepaths
    if filepaths_arr.ndim == 1:
        AMPOs = []
        for filepath in filepaths_arr:
            try:
                data = pd.read_csv(filepath).T.to_numpy()
                time, lmtc, stim, fsee, *_ = data
                AMPO = -np.trapz(fsee, lmtc) / time[-1]
                AMPOs.append(AMPO)
            except Exception:
                AMPOs.append(np.nan)
        return np.array(AMPOs)

    # Case 2: 2D list of filepaths
    elif filepaths_arr.ndim == 2:
        shape = filepaths_arr.shape
        AMPOs = np.full(shape, np.nan, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                try:
                    data = pd.read_csv(filepaths_arr[i, j]).T.to_numpy()
                    time, lmtc, stim, fsee, *_ = data
                    AMPO = -np.trapz(fsee, lmtc) / time[-1]
                    AMPOs[i, j] = AMPO
                except Exception:
                    continue  # Already NaN
        return AMPOs
    
    # Case 3: 3D list of filepaths
    elif filepaths_arr.ndim == 3:
        shape = filepaths_arr.shape
        AMPOs = np.full(shape, np.nan, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    try:
                        data = pd.read_csv(filepaths_arr[i, j, k]).T.to_numpy()
                        time, lmtc, stim, fsee, *_ = data
                        AMPO = -np.trapz(fsee, lmtc) / time[-1]
                        AMPOs[i, j, k] = AMPO
                    except Exception:
                        continue  # Already NaN
        return AMPOs
    else:
        raise ValueError("Unsupported input structure for 'filepaths'.")
        
        
