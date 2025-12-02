# minigst (Python)

The companion Python package for gstlearn.

This Python package wraps gstlearn functions to offer access to some basic geostatistical methods (mainly variography and kriging).

## Installation

### Prerequisites

You need Python 3.8 or higher. You will also need to install the gstlearn Python package. Please refer to the [gstlearn documentation](https://gstlearn.org) for installation instructions.


### Installing minigst from our pypi (official release)

You can install the official release of minigst package using pip:

```bash
cd python
pip install minigst
```


### Installing minigst from sources

After cloning the github repo, you can install the minigst package from source using pip:

```bash
git clone https://github.com/gstlearn/minigst
cd minigst
cd python
pip install .
```

Or for development:

```bash
git clone https://github.com/gstlearn/minigst
cd minigst
cd python
pip install -e .
```

## Usage

```python
import minigst as mg
import gstlearn as gl
import pandas as pd

# Load data from a pandas DataFrame
df = pd.read_csv("data.csv")
db = mg.df_to_db(df, coord_names=["x", "y"])

# Compute experimental variogram
vario_exp = mg.vario_exp(db, vname="variable", nlag=20, dlag=10.0)

# Fit a model
model = mg.model_fit(vario_exp, struct=["NUGGET", "SPHERICAL"])

# Perform kriging
target_db = mg.create_db_grid(nx=[100, 100], dx=[1.0, 1.0])
mg.minikriging(db, target_db, vname="variable", model=model)

# Plot results
mg.dbplot_grid(target_db, color="K.variable.estim")
```

## Features

The minigst Python package provides wrapper functions for:

- **Database operations**: Convert pandas DataFrames to gstlearn Db objects, create grids, manipulate variables
- **Plotting**: Visualize spatial data and grids using matplotlib
- **Variography**: Compute experimental variograms and fit models
- **Kriging**: Perform simple, ordinary, and universal kriging
- **Simulation**: Generate Gaussian random fields

## Documentation

For more information about the underlying gstlearn library, please visit [gstlearn.org](https://gstlearn.org).

## License

This package is distributed under the BSD-3 license.
