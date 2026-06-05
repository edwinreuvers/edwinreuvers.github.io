# -*- coding: utf-8 -*-
"""
This module provides small statistical and formatting helper functions.

Functions
---------
ceil(a, precision=0)
    Round values up to a given decimal precision.

floor(a, precision=0)
    Round values down to a given decimal precision.

rmse(x, y)
    Compute the root mean square error between two arrays.

pdiff(x, y)
    Compute the percentage difference of `x` relative to `y`.

zscore(x, axis=-1)
    Compute z-score normalisation along a given axis.

str_round(value, n)
    Round a value to a fixed number of significant digits and return a string.

find_max(data, grid)
    Find the maximum value in an array and return its grid coordinates.

analyse_3similar(lst, tolerance=1.0)
    Compare three values and identify whether two or more are similar.
"""

import numpy as np

def ceil(a, precision=0):
    """
    Ceil a number or array to the given decimal precision.

    Parameters
    ----------
    a : float or array-like
        Input value(s).
    precision : int, optional
        Number of decimal places. Default is 0.

    Returns
    -------
    float or numpy.ndarray
        Value(s) with ceiling applied at the requested precision.
    """
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def floor(a, precision=0):
    """
    Floor a number or array to the given decimal precision.

    Parameters
    ----------
    a : float or array-like
        Input value(s).
    precision : int, optional
        Number of decimal places. Default is 0.

    Returns
    -------
    float or numpy.ndarray
        Value(s) with floor applied at the requested precision.
    """
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def rmse(x, y):
    """
    Compute the root mean square error between two arrays.

    Parameters
    ----------
    x, y : array-like
        Input arrays with compatible shapes.

    Returns
    -------
    float
        Root mean square error.
    """
    mse = np.mean((x - y) ** 2)
    return float(np.sqrt(mse))

def pdiff(x,y):
    """
    Compute the percentage difference of `x` relative to `y`.

    Parameters
    ----------
    x, y : array-like or scalar
        Values with compatible shapes.

    Returns
    -------
    float or numpy.ndarray
        Percentage difference, computed as `(x - y) / y * 100`.
    """
    return ((x-y)/y)*100

def zscore(x, axis=-1):
    """
    Compute the z-score normalization along a given axis.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to compute the mean and std. Default is -1.

    Returns
    -------
    np.ndarray
        Z-score normalized array.
    """
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / s

def str_round(value: float, n: int) -> str:
    """
    Round a number to n significant digits and return a string representation.

    Parameters
    ----------
    value : float
        Value to round.
    n : int
        Number of significant digits.

    Returns
    -------
    str
        Rounded value formatted as a string. Returns '-' for NaN.
    """
    
    if np.isnan(value):
        return '-'
    elif value == 0:
        return '0'
    
    # Determine the number of digits before the decimal point
    num_digits = int(np.floor(np.log10(abs(value)))) + 1
    
    # Calculate the decimal places to round
    decimal_places = max(n - num_digits, 0)
    
    rounded_value = round(value, decimal_places)
    
    # Format as string with fixed decimal places if needed
    if decimal_places > 0:
        return f"{rounded_value:.{decimal_places}f}"
    else:
        return str(int(rounded_value))
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_max(data, grid):
    """
    Find the maximum value in an N-dimensional array and its corresponding values 
    from the provided grid arrays.

    Parameters
    ----------
    data : ndarray
        N-dimensional array containing the data.
    grid : tuple of ndarray
        Tuple of 1D arrays representing the coordinate axes for each dimension of `data`.
        Length must match the number of dimensions of `data`.

    Returns
    -------
    data_max : float
        Maximum value found in `data`.
    grid_max : tuple
        Coordinates corresponding to the maximum value, taken from `grid`.
        The order matches the order of `grid`.
    
    Notes
    -----
    Uses `np.nanargmax` to ignore NaN values in `data`.
    """
    max_idx = np.nanargmax(data)
    
    if len(grid) == 2:
        x, y = grid
        row_idx, col_idx = np.unravel_index(max_idx, data.shape)
        grid_max = (x[col_idx], y[row_idx])
        data_max = data[row_idx, col_idx]
        
    elif len(grid) == 3:
        x, y, z = grid
        row_idx, col_idx, dep_idx = np.unravel_index(max_idx, data.shape)
        grid_max = (x[row_idx], y[col_idx], z[dep_idx])
        data_max = data[row_idx, col_idx, dep_idx]
        
    else:
        raise ValueError("Grid must have length 2 or 3 corresponding to data dimensions.")
    
    return data_max, grid_max
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def analyse_3similar(lst, tolerance=1.0):
    """
    Analyze a list of three values to determine similarity within a tolerance.

    The function compares up to three values and identifies whether any values
    are approximately equal within a specified tolerance. It handles missing
    values (`None` or `np.nan`) and returns the most representative value,
    a differing value (if any), and an indicator of similarity.

    Parameters
    ----------
    lst : list of float or None
        A list containing exactly three elements. Elements may be numeric,
        `None`, or `np.nan`.
    tolerance : float, optional
        The maximum absolute difference for two values to be considered
        similar. Default is 1.0.

    Returns
    -------
    most_frequent : float or None
        The value considered most representative among the inputs. If no
        conclusion can be drawn, returns None.
    different_value : float or None
        A value that differs from the most representative value. Returns
        None if all valid values are similar or insufficient data exists.
    flag : bool or int
        - If bool:
            * True  -> a consistent or usable result was found
            * False -> no meaningful similarity detected
        - If int:
            Index (0, 1, or 2) of the value considered different when
            exactly one value deviates.

    Raises
    ------
    ValueError
        If `lst` does not contain exactly three elements.

    Notes
    -----
    - Missing values (`None` or `np.nan`) are ignored in comparisons.
    - If only one valid value is present, it is returned as the most
      representative value.
    - If two valid values are present, they are compared directly.
    - If all three values are valid, pairwise comparisons determine whether
      a majority agreement exists.

    Examples
    --------
    >>> analyse_3similar([1.0, 1.1, 0.9], tolerance=0.2)
    (1.0, None, True)

    >>> analyse_3similar([1.0, 5.0, 1.1], tolerance=0.2)
    (1.0, 5.0, 1)

    >>> analyse_3similar([None, 2.0, np.nan])
    (2.0, None, True)

    >>> analyse_3similar([1.0, 2.0, 3.0])
    (None, None, False)
    """
    
    if len(lst) != 3:
        raise ValueError("List must have exactly 3 items.")
    
    # Store values with their indices
    valid = [(i, v) for i, v in enumerate(lst) if not (v is None or isinstance(v, float) and np.isnan(v))]
    
    if len(valid) == 0:
        return None, None, False  # No valid data
    if len(valid) == 1:
        return valid[0][1], None, True  # Only one valid value, treat as "most frequent"

    def close(x, y):
        return abs(x - y) <= tolerance

    # Unpack valid items
    (i1, v1), (i2, v2) = valid[0], valid[1]
    
    # If only two valid values
    if len(valid) == 2:
        if close(v1, v2):
            return v1, None, True
        else:
            return v1, v2, i2  # Arbitrarily say v1 is "most frequent"

    # All three are valid
    a, b, c = lst
    if close(a, b) and close(a, c) and close(b, c):
        return a, None, True
    if close(a, b):
        return a, c, 2
    elif close(a, c):
        return a, b, 1
    elif close(b, c):
        return b, a, 0

    return None, None, False
