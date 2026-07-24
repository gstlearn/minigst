# minigst Python API Reference

This document provides an overview of the main functions available in the minigst Python package.

## Database Operations (`minigst.db`)

### `df_to_db(df, coord_names, is_grid=False)`

Create a gstlearn Db object from a pandas DataFrame.

**Parameters:**
- `df`: pandas DataFrame containing the data
- `coord_names`: List of column names to use as coordinates
- `is_grid`: Boolean, if True creates a DbGrid instead of Db

**Returns:** gstlearn Db or DbGrid object

**Example:**
```python
import pandas as pd
import minigst as mg

df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "z": [1, 2, 3]})
db = mg.df_to_db(df, coord_names=["x", "y"])
```

### `df_to_db_grid(df, coord_names)`

Create a gstlearn DbGrid object from a pandas DataFrame.

**Parameters:**
- `df`: pandas DataFrame containing the grid data
- `coord_names`: List of column names to use as coordinates

**Returns:** gstlearn DbGrid object

### `create_db_grid(coords=None, nx=None, dx=None, x0=None, coord_names=None)`

Create an empty DbGrid.

**Parameters:**
- `coords`: List of coordinate arrays for each dimension
- `nx`: List of grid sizes in each dimension
- `dx`: List of grid steps in each dimension
- `x0`: List of grid origins in each dimension
- `coord_names`: List of coordinate names

**Returns:** gstlearn DbGrid object

**Example:**
```python
import numpy as np
import minigst as mg

# Using coordinate arrays
xseq = np.linspace(0, 1, 100)
yseq = np.linspace(0, 1, 100)
db_grid = mg.create_db_grid(coords=[xseq, yseq], coord_names=["x", "y"])

# Using grid parameters
db_grid = mg.create_db_grid(nx=[100, 100], dx=[0.01, 0.01], x0=[0, 0])
```

### `add_var_to_db(db, var, vname)`

Add variable(s) to a Db object.

**Parameters:**
- `db`: gstlearn Db object
- `var`: Array or matrix of values to add
- `vname`: Name(s) of the variable(s) to add

### `del_var_from_db(db, vname)`

Delete variable(s) from a Db object.

**Parameters:**
- `db`: gstlearn Db object
- `vname`: Name(s) of the variable(s) to delete (string or list)

### `summary_stats(db, vname, stat=None, only_common=False)`

Compute summary statistics for variables in a Db.

**Parameters:**
- `db`: gstlearn Db object
- `vname`: Variable name(s) (string or list)
- `stat`: List of statistics to compute (default: all)
- `only_common`: If True, only use rows without any NA values

**Returns:** pandas DataFrame with statistics

## Plotting (`minigst.plot`)

### `dbplot_point(db, color=None, size=None, cmap=None, ...)`

Plot variables in a Db as isolated points.

**Parameters:**
- `db`: gstlearn Db object
- `color`: Name of variable for coloring points
- `size`: Name of variable for sizing points
- `cmap`: Colormap name (default: 'viridis')
- `size_range`: Tuple specifying min and max point sizes (default: (0.5, 3))
- `aspect`: Aspect ratio of the plot
- `xlabel`, `ylabel`, `title`: Plot labels
- `ax`: Matplotlib axis object (if None, creates new figure)

**Returns:** Matplotlib axis object

### `dbplot_grid(db_grid, color=None, cmap=None, ...)`

Plot a variable on a DbGrid.

**Parameters:**
- `db_grid`: gstlearn DbGrid object
- `color`: Name of the variable to plot
- `cmap`: Colormap name (default: 'viridis')
- `aspect`: Aspect ratio
- `xlabel`, `ylabel`, `title`: Plot labels
- `ax`: Matplotlib axis object

**Returns:** Matplotlib axis object

### `add_points(x, y, ax=None, color='black', marker='o', size=50, **kwargs)`

Add points to an existing plot.

### `add_lines(ax=None, v=None, h=None, a=None, b=None, ...)`

Add lines to an existing plot.

**Parameters:**
- `ax`: Matplotlib axis object
- `v`: X-coordinate for vertical line
- `h`: Y-coordinate for horizontal line
- `a`, `b`: Slope and intercept for line y = a*x + b
- `color`, `linestyle`, `linewidth`: Line properties

## Variography (`minigst.vario`)

### `vario_exp(db, vname, pol_drift=None, ext_drift=None, dir=None, nlag=20, dlag=100, ...)`

Compute an experimental variogram.

**Parameters:**
- `db`: gstlearn Db object
- `vname`: Variable name(s)
- `pol_drift`: Order of polynomial drift
- `ext_drift`: External drift variable name(s)
- `dir`: Direction (angle in degrees for 2D, or direction vectors)
- `nlag`: Number of lags
- `dlag`: Distance of lags
- `toldis`: Tolerance on distance (0 to 1)
- `tolang`: Tolerance on angle (0 to 90 degrees)

**Returns:** gstlearn Vario object

### `create_model(struct, ndim=2)`

Create a variogram model.

**Parameters:**
- `struct`: Structure type(s) (string or list)
- `ndim`: Space dimension

**Returns:** gstlearn Model object

### `model_fit(vario, struct, prune_model=True, aniso_model=True)`

Fit a variogram model to experimental variogram.

**Parameters:**
- `vario`: gstlearn Vario object (experimental variogram)
- `struct`: Structure type(s) (string or list)
- `prune_model`: Boolean, if True removes low-variance components
- `aniso_model`: Boolean, if True allows anisotropy

**Returns:** gstlearn Model object

**Example:**
```python
import minigst as mg

# Compute experimental variogram
vario = mg.vario_exp(db, vname="elevation", nlag=20, dlag=10.0)

# Fit model
model = mg.model_fit(vario, struct=["NUGGET", "SPHERICAL"])
```

## Kriging (`minigst.kriging`)

### `minikriging(dbin, dbout, vname, model, type='ordinary', ...)`

Compute kriging predictions.

**Parameters:**
- `dbin`: gstlearn Db object with observation data
- `dbout`: gstlearn Db object for target predictions
- `vname`: Variable name(s) to predict
- `model`: gstlearn Model object
- `type`: Kriging type ('simple' or 'ordinary')
- `pol_drift`: Polynomial drift order for universal kriging
- `ext_drift`: External drift variable name(s)
- `mean`: Mean value for simple kriging
- `neighborhood`: Neighborhood specification ('unique' or numeric)
- `std`: Boolean, compute standard deviations
- `prefix`: Prefix for output variable names

**Example:**
```python
import minigst as mg

mg.minikriging(
    obs_db, target_db, vname="temperature", model=model, type="ordinary", std=True
)
```

### `minixvalid(dbin, vname, model, type='ordinary', ...)`

Compute kriging cross-validation.

**Parameters:** Similar to `minikriging` but operates on a single Db

### `kriging_mean(db, vname, model)`

Calculate the kriging mean of a variable.

**Parameters:**
- `db`: gstlearn Db object
- `vname`: Variable name
- `model`: gstlearn Model object

**Returns:** Float value

### `set_mean(model, mu)`

Set the mean value in a Model object.

## Simulation (`minigst.gaussim`)

### `simulate_gauss_rf(target, model, nsim=1, ntuba=100, seed=43431, ...)`

Simulate a Gaussian random field with the turning bands algorithm.

**Parameters:**
- `target`: gstlearn Db object for target locations
- `model`: gstlearn Model object
- `nsim`: Number of simulations to generate
- `ntuba`: Number of turning bands to use
- `seed`: Random seed for reproducibility
- `simname`: Optional name for the simulated variable
- `dbcond`: Optional Db object with conditioning data
- `vcond`: Optional variable name in dbcond for conditioning

**Example:**
```python
import minigst as mg

# Unconditional simulation
mg.simulate_gauss_rf(target_db, model, nsim=5, seed=12345)

# Conditional simulation
mg.simulate_gauss_rf(target_db, model, nsim=5, dbcond=obs_db, vcond="observations")
```

## Available Variogram Structures

Common structure types that can be used in `create_model` and `model_fit`:

- `'NUGGET'`: Nugget effect
- `'SPHERICAL'`: Spherical model
- `'EXPONENTIAL'`: Exponential model
- `'GAUSSIAN'`: Gaussian model
- `'CUBIC'`: Cubic model
- `'MATERN'`: Matérn model
- `'K_BESSEL'`: K-Bessel model

For a complete list, refer to the [gstlearn documentation](https://gstlearn.org).
