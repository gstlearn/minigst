"""Variogram functions for minigst package."""

import numpy as np
import gstlearn as gl

from .plot import dbplot_grid


def _rep_arg(n, arg, argname):
    """Adapt the size of model parameters."""
    if not isinstance(arg, (list, tuple, np.ndarray)):
        return [arg] * max(1, n)
    elif n == 0:
        raise ValueError(f"The length of the argument {argname} should be 1.")
    elif len(arg) != n and len(arg) != 1:
        raise ValueError(
            f"The length of the argument {argname} should be either 1 or {n}."
        )
    elif len(arg) == 1:
        return list(arg) * n
    else:
        return list(arg)


def vario_exp(
    db,
    vname,
    pol_drift=None,
    ext_drift=None,
    dir=None,
    nlag=20,
    dlag=100,
    toldis=0.5,
    tolang=None,
):
    """
    Compute an experimental variogram.

    Args:
        db: gstlearn Db object
        vname: Variable name(s) (string or list of strings)
        pol_drift: Order of polynomial drift for universal kriging
        ext_drift: External drift variable name(s)
        dir: Direction for variogram computation (angle in degrees for 2D, or direction vectors)
        nlag: Number of lags
        dlag: Distance of lags
        toldis: Tolerance on distance (0 to 1)
        tolang: Tolerance on angle (0 to 90 degrees)

    Returns:
        gstlearn Vario object containing experimental variogram

    Examples:
        >>> import minigst as mg
        >>> vario = mg.vario_exp(db, vname='elevation', nlag=20, dlag=10.0)
        >>> # Directional variogram
        >>> vario = mg.vario_exp(db, vname='elevation', dir=[30, -30], nlag=20, dlag=10.0)
    """
    model = None
    # Set variable as Z locator
    if isinstance(vname, str):
        vname = [vname]

    db.setLocators(vname, gl.ELoc.Z, cleanSameLocator=True)

    # Handle drifts
    if ext_drift is not None or pol_drift is not None:
        if pol_drift is not None:
            for i in range(pol_drift + 1):
                # Add polynomial drift terms (simplified)
                pass
        else:
            pol_drift = 0
        if ext_drift is not None:
            if isinstance(ext_drift, str):
                ext_drift = [ext_drift]
            db.setLocators(ext_drift, gl.ELoc.F)
        model = gl.Model.createFromParam()
        ndrifts = db.getNLoc(gl.ELoc.F)
        model.setDriftIRF(pol_drift, ndrifts)
    # Create variogram parameters
    if dir is None:
        # Omnidirectional variogram
        vario_param = gl.VarioParam.createOmniDirection(
            nlag=nlag, dlag=dlag, toldis=toldis
        )
    else:
        # Directional variogram
        if not isinstance(dir, (list, tuple, np.ndarray)):
            dir = [dir]

        if isinstance(dir[0], (int, float)):
            # Angles in 2D
            angles = True
            ndir = len(dir)
        else:
            # Direction vectors
            angles = False
            dir = np.array(dir)
            ndir = dir.shape[0]

        # Replicate parameters
        nlag = _rep_arg(ndir, nlag, "nlag")
        dlag = _rep_arg(ndir, dlag, "dlag")
        toldis = _rep_arg(ndir, toldis, "toldis")

        if tolang is None:
            tolang = [180 / (2 * ndir)] * ndir
        else:
            tolang = _rep_arg(ndir, tolang, "tolang")

        vario_param = gl.VarioParam()
        for i in range(ndir):
            if angles:
                dir_param = gl.DirParam.create(
                    nlag=nlag[i],
                    dlag=dlag[i],
                    toldis=toldis[i],
                    tolang=tolang[i],
                    angle2D=dir[i],
                )
            else:
                dir_param = gl.DirParam.create(
                    nlag=nlag[i],
                    dlag=dlag[i],
                    toldis=toldis[i],
                    tolang=tolang[i],
                    codir=dir[i],
                )
            vario_param.addDir(dir_param)

    # Compute variogram
    vario = gl.Vario(vario_param)
    vario.compute(db, model=model)

    return vario


def vario_map(db, vname, grid_res=20):
    """
    Compute a variogram map for a variable in a gstlearn Db object.

    Args:
        db: gstlearn Db object
        vname: Name of the variable (string)
        grid_res: Grid resolution (int or float). Defines the number of cells
                  in each direction for the computation grid.
        plot: If True, plot the resulting variogram maps.

    Returns:
        DbGrid object containing the computed variogram map.

    Details:
        The variogram map is computed on a 2D grid centered at the origin,
        with dimensions (2 * grid_res + 1) × (2 * grid_res + 1).

    Examples:
        >>> import minigst as mg
        >>> Scotland, _ = data("Scotland")
        >>> db = mg.df_to_db(df=Scotland, coord_names=["Longitude", "Latitude"])
        >>> vario_map_grid = mg.vario_map(db=db, vname="Elevation", plot=True)
    """
    plot = False
    # Set the variable to be analyzed
    db.setLocator(vname, gl.ELoc.Z, cleanSameLocator=True)

    # Check argument type
    if not isinstance(grid_res, (int, float)):
        raise TypeError("grid_res must be numeric")

    # Compute variogram map grid
    # Equivalent de : db_vmap(db, nxx = rep(gridRes[1], db$getNDim()))
    n_dim = db.getNDim()  # TODO: adapter selon gstlearn
    grid_vmap = gl.db_vmap(
        db, nxx=np.repeat(grid_res, n_dim)
    )  # TODO: remplacer par appel correct

    if plot:
        # Extract variable names for plotting
        vn_var = grid_vmap.get_names("VMAP.*.Var")[0]
        vn_nb = grid_vmap.get_names("VMAP.*.Nb")[0]

        # Plot variogram map (variance)
        p1 = dbplot_grid(
            grid_vmap,
            color=vn_var,
            cmap="Spectral",
            legend_title="Var",
            title=f"Variogram map : {vname}",
        )

        # Plot number of pairs
        p2 = dbplot_grid(
            grid_vmap,
            color=vn_nb,
            cmap="RdBu",
            legend_title="Nb",
            title="Number of pairs",
        )

        # Display side-by-side
        from matplotlib import pyplot as plt

        _, axes = plt.subplots(1, 2, figsize=(10, 4))
        p1(axes[0])
        p2(axes[1])
        plt.tight_layout()
        plt.show()

    return grid_vmap
