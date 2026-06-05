"""
This module provides interpolation functions for structured data grids.

The functions interpolate 1D curves, 2D grids and 3D grids onto finer grids.
Missing values (NaNs) are removed before interpolation.

Functions
---------
do_2d(x, y, **kwargs)
    Interpolate 1D data (x, y) onto a finer grid.

do_3d(data, grid, **kwargs)
    Interpolate 2D grid data onto a finer 2D grid.

do_4d(data, grid, **kwargs)
    Interpolate 3D grid data onto a finer 3D grid.
"""

import numpy as np
from scipy.interpolate import interp1d, griddata

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_2d(x, y, **kwargs):
    """
    Interpolate 1D data onto a finer grid.

    Parameters
    ----------
    x : array-like
        1D array of x-coordinates.
    y : array-like
        1D array of y-values corresponding to `x`.
    **kwargs : dict, optional
        Additional keyword arguments:
        
        N : int, optional
            Number of interpolation points (default is 100).
        method : str, optional
            Interpolation method passed to `scipy.interpolate.interp1d`
            (default is 'cubic').

    Returns
    -------
    x_fine : ndarray
        Interpolated x-coordinates.
    y_fine : ndarray
        Interpolated y-values.

    Notes
    -----
    NaN values in `y` are removed prior to interpolation.
    """
    num_points = kwargs.get("N", 100)
    method = kwargs.get("method", "cubic")

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    valid_mask = ~np.isnan(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Fine grid
    x_fine = np.linspace(x[0], x[-1], num_points)

    interpolator = interp1d(
        x_valid,
        y_valid,
        kind=method,
        bounds_error=False,
    )
    y_fine = interpolator(x_fine)

    return x_fine, y_fine

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_3d(data, grid, **kwargs):
    """
    Interpolate 2D grid data onto a finer grid.

    Parameters
    ----------
    data : ndarray
        2D array of data values.
    grid : tuple of array-like
        Tuple (x, y) defining the grid coordinates.
    **kwargs : dict, optional
        Additional keyword arguments:
        
        N : int, optional
            Number of interpolation points along x (default is 100).
        method : str, optional
            Interpolation method for `scipy.interpolate.griddata`
            (default is 'linear').

    Returns
    -------
    data_fine : ndarray
        Interpolated data on the finer grid.
    grid_fine : tuple of ndarray
        Tuple (x_fine, y_fine) defining the new grid.

    Notes
    -----
    NaN values in `data` are removed prior to interpolation.
    """
    num_points = kwargs.get("N", 100)
    method = kwargs.get("method", "linear")

    x, y = grid

    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)

    # Meshgrid
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

    # Flatten
    x_flat = x_mesh.ravel()
    y_flat = y_mesh.ravel()
    data_flat = data.ravel()

    # Remove NaNs
    valid_mask = ~np.isnan(data_flat)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    data_valid = data_flat[valid_mask]

    # Fine grid
    x_fine = np.linspace(x[0], x[-1], num_points)
    y_fine = np.linspace(y[0], y[-1], num_points + 1)
    x_fine_mesh, y_fine_mesh = np.meshgrid(x_fine, y_fine, indexing="ij")

    interpolated_values = griddata(
        points=np.vstack([x_valid, y_valid]).T,
        values=data_valid,
        xi=np.vstack([x_fine_mesh.ravel(), y_fine_mesh.ravel()]).T,
        method=method,
    )

    data_fine = interpolated_values.reshape((num_points, num_points + 1))
    grid_fine = (x_fine, y_fine)

    return data_fine.T, grid_fine

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def do_4d(data, grid, **kwargs):
    """
    Interpolate 3D grid data onto a finer grid.

    Parameters
    ----------
    data : ndarray
        3D array of data values.
    grid : tuple of array-like
        Tuple (x, y, z) defining the grid coordinates.
    **kwargs : dict, optional
        Additional keyword arguments:
        
        N : int, optional
            Number of interpolation points along x (default is 100).
        method : str, optional
            Interpolation method for `scipy.interpolate.griddata`
            (default is 'linear').

    Returns
    -------
    data_fine : ndarray
        Interpolated 3D data.
    grid_fine : tuple of ndarray
        Tuple (x_fine, y_fine, z_fine) defining the new grid.

    Notes
    -----
    NaN values in `data` are removed prior to interpolation.
    """
    num_points = kwargs.get("N", 100)
    method = kwargs.get("method", "linear")

    x, y, z = grid

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    data = np.asarray(data)

    # Meshgrid
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z, indexing="ij")

    # Flatten
    x_flat = x_mesh.ravel()
    y_flat = y_mesh.ravel()
    z_flat = z_mesh.ravel()
    data_flat = data.ravel()

    # Remove NaNs
    valid_mask = ~np.isnan(data_flat)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    data_valid = data_flat[valid_mask]

    # Fine grid
    x_fine = np.linspace(x[0], x[-1], num_points)
    y_fine = np.linspace(y[0], y[-1], num_points + 1)
    z_fine = np.linspace(z[0], z[-1], num_points + 2)

    x_fine_mesh, y_fine_mesh, z_fine_mesh = np.meshgrid(
        x_fine, y_fine, z_fine, indexing="ij"
    )

    interpolated_values = griddata(
        points=np.vstack([x_valid, y_valid, z_valid]).T,
        values=data_valid,
        xi=np.vstack(
            [
                x_fine_mesh.ravel(),
                y_fine_mesh.ravel(),
                z_fine_mesh.ravel(),
            ]
        ).T,
        method=method,
    )

    data_fine = interpolated_values.reshape(
        (num_points, num_points + 1, num_points + 2)
    )
    grid_fine = (x_fine, y_fine, z_fine)

    return data_fine, grid_fine
