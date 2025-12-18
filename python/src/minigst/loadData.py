import gstlearn as gl
import pandas as pd

import importlib.resources as pkg_resources
from . import datafiles


def data(name):
    """
    Return data contained in the package

    Args:
        name: name of the data set

    Returns:
        various objects according to the case.

    Examples:
        >>> import minigst as mg
        >>> dat, grid = mg.data("Scotland") # 2 pandas DataFrame
        >>> dbgrid = mg.data("ScotlandGrid") # a gstlearn.DbGrid object
    """
    if name == "Scotland":
        file_path = (
            pkg_resources.files(datafiles) / "Scotland" / "Scotland_Temperatures.csv"
        )
        scot = pd.read_csv(str(file_path), na_values="MISS")
        file_path = (
            pkg_resources.files(datafiles) / "Scotland" / "Scotland_Elevations.csv"
        )
        scot_grid = pd.read_csv(str(file_path))
        return scot, scot_grid
    if name == "ScotlandGrid":
        file_path = (
            pkg_resources.files(datafiles) / "Scotland" / "Scotland_Elevations.NF"
        )
        scot_grid = gl.DbGrid.createFromNF(str(file_path))
        return scot_grid
    if name == "Meuse":
        file_path = pkg_resources.files(datafiles) / "Meuse" / "meuse.csv"
        meuse = pd.read_csv(str(file_path))
        file_path = pkg_resources.files(datafiles) / "Meuse" / "meuse.grid.csv"
        meuse_grid = pd.read_csv(str(file_path))
        return meuse, meuse_grid
    if name == "Jura":
        file_path = pkg_resources.files(datafiles) / "Jura" / "jura_val_loc.csv"
        jura_val_loc = pd.read_csv(str(file_path))
        file_path = pkg_resources.files(datafiles) / "Jura" / "jura_grid.csv"
        jura_grid = pd.read_csv(str(file_path))
        file_path = pkg_resources.files(datafiles) / "Jura" / "jura_pred.csv"
        jura_pred = pd.read_csv(str(file_path))
        return jura_pred, jura_grid, jura_val_loc
    print("No data named " + name + " in the minigst package")
