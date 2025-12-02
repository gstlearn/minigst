# minigst (R)

The companion R package for gstlearn.

This R package wraps gstlearn functions to offer access to some basic geostatistical methods (mainly variography and kriging).

## Installation

### Prerequisites

You need R 4.0 or higher. You will also need to install the gstlearn R package. Please refer to the [gstlearn documentation](https://gstlearn.org) for installation instructions.

### Installing minigst from our CRAN (official release)

You can install the official release of minigst package using the following R command:

```R
install.packages("minigst", repos="https://soft.mines-paristech.fr/cran")
```

### Installing minigst from source

After cloning the github repo, you can install the minigst package from source using the following shell command:

```bash
git clone https://github.com/gstlearn/minigst
cd minigst
cd R
R CMD INSTALL .
```


## Usage

```R
library(minigst)

# Load data from a pandas DataFrame
df = read.csv("data.csv")
db = dfToDb(df, coord_names=["x", "y"])

# Compute experimental variogram
vario_exp = vario_exp(db, vname="variable", nlag=20, dlag=10.0)

# Fit a model
model = model_fit(vario_exp, struct=["NUGGET", "SPHERICAL"])

# Perform kriging
target_db = createDbGrid(nx=[100, 100], dx=[1.0, 1.0])
minikriging(db, target_db, vname="variable", model=model)

# Plot results
dbplot_grid(target_db, color="K.variable.estim")
```

## Features

The minigst R package provides wrapper functions for:

- **Database operations**: Convert pandas DataFrames to gstlearn Db objects, create grids, manipulate variables
- **Plotting**: Visualize spatial data and grids using matplotlib
- **Variography**: Compute experimental variograms and fit models
- **Kriging**: Perform simple, ordinary, and universal kriging
- **Simulation**: Generate Gaussian random fields

## Documentation

For more information about the underlying gstlearn library, please visit [gstlearn.org](https://gstlearn.org).

## License

This package is distributed under the GPL license.
