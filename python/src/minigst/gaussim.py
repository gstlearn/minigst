"""Gaussian simulation functions for minigst package."""

import numpy as np
import gstlearn as gl
from .kriging import kriging_mean, set_mean


def simulate_gauss_rf(
    target, model, nsim=1, ntuba=1000, seed=43431, simname=None, dbcond=None, vcond=None
):
    """
    Simulate a Gaussian random field with the turning bands algorithm.

    Args:
        target: gstlearn Db object for target locations
        model: gstlearn Model object defining the covariance structure
        nsim: Number of simulations to generate
        ntuba: Number of turning bands to use
        seed: Random seed for reproducibility
        simname: Optional name for the simulated variable
        dbcond: Optional Db object with conditioning data
        vcond: Optional variable name in dbcond for conditioning

    Returns:
        None (results added directly to target)

    Examples:
        >>> import minigst as mg
        >>> # Unconditional simulation
        >>> mg.simulate_gauss_rf(target_db, model, nsim=5, seed=12345)
        >>>
        >>> # Conditional simulation
        >>> mg.simulate_gauss_rf(target_db, model, nsim=5,
        ...                      dbcond=obs_db, vcond='observations')
    """
    # Set random seed
    gl.law_set_random_seed(seed)

    # Set naming convention
    if simname is not None:
        namconv = gl.NamingConvention(simname)
    else:
        namconv = gl.NamingConvention("Simu")

    if dbcond is not None and vcond is not None:
        # Conditional simulation
        if isinstance(vcond, str):
            dbcond.setLocator(vcond, gl.ELoc.Z)

        model_copy = model.clone()
        set_mean(model_copy, kriging_mean(dbcond, vcond, model))
        # Perform conditional simulation
        err = gl.simtub(
            dbcond,
            target,
            model_copy,
            neigh=gl.NeighUnique(),
            nbtuba=ntuba,
            nbsimu=nsim,
            namconv=namconv,
        )
    else:
        # Unconditional simulation
        err = gl.simtub(None, target, model, nbtuba=ntuba, nbsimu=nsim, namconv=namconv)

    target.clearLocators(gl.ELoc.Z)

    if err != 0:
        raise RuntimeError("Simulation failed with error code: {}".format(err))
