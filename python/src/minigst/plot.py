"""Plotting functions for minigst package."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import gstlearn as gl


def dbplot_point(
    db,
    color=None,
    size=None,
    cmap=None,
    size_range=(0.5, 3),
    aspect=1,
    xlabel=None,
    ylabel=None,
    title=None,
    ax=None,
    **kwargs,
):
    """
    Plot variables in a Db as isolated points.

    Args:
        db: gstlearn Db object
        color: Name of the variable for coloring points
        size: Name of the variable for sizing points
        cmap: Colormap name (default: 'viridis')
        size_range: Tuple specifying min and max point sizes
        aspect: Aspect ratio of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        ax: Matplotlib axis object (if None, creates new figure)
        **kwargs: Additional arguments passed to scatter

    Returns:
        Matplotlib axis object

    Examples:
        >>> import minigst as mg
        >>> ax = mg.dbplot_point(db, color='elevation', size='temperature')
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get coordinates
    coord_names = db.getNamesByLocator(gl.ELoc.X)
    if len(coord_names) < 2:
        raise ValueError("Db must have at least 2 coordinate dimensions.")

    x = db[coord_names[0]]
    y = db[coord_names[1]]

    # Apply selection if it exists
    sel_names = db.getNamesByLocator(gl.ELoc.SEL)
    if len(sel_names) > 0:
        sel_values = db[sel_names[0]]
        mask = sel_values == 1
        x = x[mask]
        y = y[mask]
    else:
        mask = np.ones(len(x), dtype=bool)

    # Prepare scatter plot parameters
    scatter_kwargs = kwargs.copy()

    # Handle color
    if color is not None:
        c = db[color]
        if len(sel_names) > 0:
            c = c[mask]
        scatter_kwargs["c"] = c
        if cmap is not None:
            scatter_kwargs["cmap"] = cmap
        else:
            scatter_kwargs["cmap"] = "viridis"

    # Handle size
    if size is not None:
        s = db[size]
        if len(sel_names) > 0:
            s = s[mask]
        # Normalize sizes to size_range
        s_min, s_max = np.nanmin(s), np.nanmax(s)
        if s_max > s_min:
            s_norm = (s - s_min) / (s_max - s_min)
            s_scaled = size_range[0] + s_norm * (size_range[1] - size_range[0])
            scatter_kwargs["s"] = s_scaled * 100  # Scale for matplotlib
        else:
            scatter_kwargs["s"] = np.mean(size_range) * 100

    # Create scatter plot
    scatter = ax.scatter(x, y, **scatter_kwargs)

    # Add colorbar if color is specified
    if color is not None:
        plt.colorbar(scatter, ax=ax, label=color)

    # Set labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(coord_names[0])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(coord_names[1])

    if title is not None:
        ax.set_title(title)

    if aspect == 1:
        ax.set_aspect("equal")

    return ax


def dbplot_grid(
    db_grid,
    color=None,
    cmap=None,
    aspect=1,
    xlabel=None,
    ylabel=None,
    title=None,
    ax=None,
    **kwargs,
):
    """
    Plot a variable on a DbGrid.

    Args:
        db_grid: gstlearn DbGrid object
        color: Name of the variable to plot
        cmap: Colormap name (default: 'viridis')
        aspect: Aspect ratio of the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title
        ax: Matplotlib axis object (if None, creates new figure)
        **kwargs: Additional arguments passed to imshow

    Returns:
        Matplotlib axis object

    Examples:
        >>> import minigst as mg
        >>> ax = mg.dbplot_grid(db_grid, color='predictions')
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get grid dimensions
    nx = db_grid.getNXs()
    if len(nx) != 2:
        raise ValueError("dbplot_grid only supports 2D grids.")

    # Get coordinates
    coord_names = db_grid.getNamesByLocator(gl.ELoc.X)
    x0 = db_grid.getX0s()
    dx = db_grid.getDXs()

    # Get data
    if color is not None:
        data = db_grid[color]
        data = data.reshape(nx[1], nx[0])  # Reshape to grid
    else:
        raise ValueError("color parameter must be specified.")

    # Create extent for imshow
    extent = [x0[0], x0[0] + nx[0] * dx[0], x0[1], x0[1] + nx[1] * dx[1]]

    # Plot
    if cmap is None:
        cmap = "viridis"

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="auto" if aspect != 1 else "equal",
        **kwargs,
    )

    # Add colorbar
    plt.colorbar(im, ax=ax, label=color if color else "")

    # Set labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(coord_names[0] if len(coord_names) > 0 else "x")

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(coord_names[1] if len(coord_names) > 1 else "y")

    if title is not None:
        ax.set_title(title)

    return ax


def add_lines(ax=None, v=None, h=None, a=None, b=None, **kwargs):
    """
    Add lines to an existing plot.

    Args:
        ax: Matplotlib axis object (if None, uses current axis)
        v: X-coordinate for vertical line
        h: Y-coordinate for horizontal line
        a: Slope for line y = a*x + b
        b: Intercept for line y = a*x + b
        **kwargs: Additional arguments passed to plot functions

    Returns:
        Matplotlib axis object

    Examples:
        >>> import minigst as mg
        >>>
        >>> ax = mg.dbplot_point(db, color='elevation')
        >>> mg.add_lines(ax=ax, v=300, c='red')
        >>> mg.add_lines(ax=ax, a=2, b=100, c='blue', linestyle='--')
        >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if v is not None:
        ax.axvline(x=v, **kwargs)

    if h is not None:
        ax.axhline(y=h, **kwargs)

    if a is not None and b is not None:
        xlim = ax.get_xlim()
        x_vals = np.array(xlim)
        y_vals = a * x_vals + b
        ax.plot(x_vals, y_vals, **kwargs)

    return ax
