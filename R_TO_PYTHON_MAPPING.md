# R to Python Function Mapping

This document shows the correspondence between functions in the R and Python versions of minigst.

## Database Operations

| R Function | Python Function | Description |
|------------|-----------------|-------------|
| `dfToDb()` | `df_to_db()` | Convert DataFrame to Db |
| `dfToDbGrid()` | `df_to_db_grid()` | Convert DataFrame to DbGrid |
| `createDbGrid()` | `create_db_grid()` | Create empty DbGrid |
| `addVarToDb()` | `add_var_to_db()` | Add variables to Db |
| `delVarFromDb()` | `del_var_from_db()` | Delete variables from Db |
| `summaryStats()` | `summary_stats()` | Compute summary statistics |

## Plotting Functions

| R Function | Python Function | Description |
|------------|-----------------|-------------|
| `dbplot_point()` | `dbplot_point()` | Plot Db points |
| `dbplot_grid()` | `dbplot_grid()` | Plot DbGrid |
| `addPoints()` | `add_points()` | Add points to plot |
| `addLines()` | `add_lines()` | Add lines to plot |

## Variogram Functions

| R Function | Python Function | Description |
|------------|-----------------|-------------|
| `vario_exp()` | `vario_exp()` | Compute experimental variogram |
| `createModel()` | `create_model()` | Create variogram model |
| `model_fit()` | `model_fit()` | Fit variogram model |

## Kriging Functions

| R Function | Python Function | Description |
|------------|-----------------|-------------|
| `minikriging()` | `minikriging()` | Perform kriging |
| `minixvalid()` | `minixvalid()` | Cross-validation |
| `kriging_mean()` | `kriging_mean()` | Calculate kriging mean |
| `setMean()` | `set_mean()` | Set mean in model |

## Simulation Functions

| R Function | Python Function | Description |
|------------|-----------------|-------------|
| `simulate_gaussRF()` | `simulate_gauss_rf()` | Simulate Gaussian random field |

## Key Differences

### Naming Convention
- R uses camelCase (e.g., `dfToDb`)
- Python uses snake_case (e.g., `df_to_db`)

### Plotting
- R version uses ggplot2 and returns ggplot objects
- Python version uses matplotlib and returns matplotlib axis objects

### Package Dependencies
- **R**: gstlearn, ggplot2, ggpubr, ggnewscale
- **Python**: gstlearn, matplotlib, numpy, pandas

### Import Statements

**R:**
```r
library(minigst)
```

**Python:**
```python
import minigst as mg
```

## Example Code Comparison

### Creating a Db Object

**R:**
```r
library(minigst)
data(Scotland)
db <- dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
```

**Python:**
```python
import minigst as mg
import pandas as pd

df = pd.read_csv("scotland.csv")
db = mg.df_to_db(df, coord_names=["Longitude", "Latitude"])
```

### Computing Variogram and Kriging

**R:**
```r
# Compute variogram
vario <- vario_exp(db=db, vname="Elevation", nlag=20, dlag=10.)

# Fit model
model <- model_fit(vario, struct=c("NUGGET", "SPHERICAL"))

# Kriging
minikriging(obsDb, targetDb, vname="Elevation", model=model, std=TRUE)
```

**Python:**
```python
# Compute variogram
vario = mg.vario_exp(db, vname="Elevation", nlag=20, dlag=10.0)

# Fit model
model = mg.model_fit(vario, struct=["NUGGET", "SPHERICAL"])

# Kriging
mg.minikriging(obs_db, target_db, vname="Elevation", model=model, std=True)
```

### Plotting

**R:**
```r
# Create plot
plt <- dbplot_point(db=db, size="Elevation", color="Longitude")
print(plt)
```

**Python:**
```python
import matplotlib.pyplot as plt

# Create plot
ax = mg.dbplot_point(db, size="Elevation", color="Longitude")
plt.show()
```
