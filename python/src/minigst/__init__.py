"""
minigst - Wrapper package for gstlearn (Python version)

This package provides wrapper functions for an easy access to gstlearn's main tools and functions.
"""

__version__ = "0.1.0"

from .db import (
    df_to_db,
    df_to_dbgrid,
    create_dbgrid,
    add_var_to_db,
    del_var_from_db,
    add_sel,
    clear_sel,
    summary_stats,
    set_var,
)

from .plot import (
    dbplot_point,
    dbplot_grid,
    add_lines,
)

from .model import (
    create_model,
    get_all_struct,
    print_all_struct,
    create_model_iso,
    model_fit,
    model_mle,
    eval_cov_matrix,
    eval_drift_matrix,
    model_compute_log_likelihood,
    set_sill,
    set_range,
)

from .vario import vario_exp, vario_map

from .kriging import (
    minikriging,
    minixvalid,
    kriging_mean,
    set_mean,
    regression,
)

from .gaussim import (
    simulate_gauss_rf,
)


from .loadData import data

__all__ = [
    "__version__",
    # Database functions
    "data",
    "df_to_db",
    "df_to_dbgrid",
    "create_dbgrid",
    "add_var_to_db",
    "del_var_from_db",
    "add_sel",
    "clear_sel",
    "summary_stats",
    "set_var",
    # Plotting functions
    "dbplot_point",
    "dbplot_grid",
    "add_lines",
    # Model functions
    "get_all_struct",
    "print_all_struct",
    "create_model_iso",
    "model_fit",
    "model_mle",
    "eval_cov_matrix",
    "eval_drift_matrix",
    "model_compute_log_likelihood",
    "set_sill",
    "set_range",
    # Variogram functions
    "vario_exp",
    "create_model",
    "vario_map",
    # Kriging functions
    "minikriging",
    "minixvalid",
    "kriging_mean",
    "set_mean",
    "regression",
    # Simulation functions
    "simulate_gauss_rf",
]
