import gstlearn as gl
import numpy as np
from .db import set_var


def create_model(struct, ndim=2, nvar=1):
    """
    Create a variogram model.

    Args:
        struct: Structure type(s) (string or list of strings)
        ndim: Space dimension
        nvar: number of variables

    Returns:
        gstlearn Model object

    Examples:
        >>> import minigst as mg
        >>> model = mg.create_model('SPHERICAL', ndim=2)
        >>> model = mg.create_model(['NUGGET', 'SPHERICAL'], ndim=2, nvar = 2)
    """
    if isinstance(struct, str):
        struct = [struct]

    context = gl.CovContext(nvar, ndim)
    model = gl.Model.create(context)

    # Add additional structures
    for s in struct:
        cov_type = gl.ECov.fromKey(s)
        model.addCovFromParam(cov_type, sills=np.identity(nvar))

    return model


def add_drifts_to_model(mdl, pol_drift=None, n_ext_drift=0, type="ordinary"):
    err = mdl.delAllDrifts()

    if pol_drift is None:
        pol_drift = -1 if type == "simple" else 0

    mdl.setDriftIRF(pol_drift, n_ext_drift)


def model_fit(vario, struct, aniso_model=True):
    """
    Fit a variogram model to experimental variogram.

    Args:
        vario: gstlearn Vario object (experimental variogram)
        struct: Structure type(s) (string or list of strings)
        aniso_model: Boolean, if True allows anisotropy

    Returns:
        gstlearn Model object

    Examples:
        >>> import minigst as mg
        >>> vario = mg.vario_exp(db, vname='elevation', nlag=20, dlag=10.0)
        >>> model = mg.model_fit(vario, struct=['NUGGET', 'SPHERICAL'])
    """
    if isinstance(struct, str):
        struct = [struct]

    ndim = vario.getNDim()
    nvar = vario.getNVar()
    # Create initial model
    model = create_model(struct, ndim=ndim, nvar=nvar)

    # Fit model

    option = gl.ModelOptimParam.create(aniso_model)
    err = model.fitNew(vario=vario, mop=option)

    # Prune model if requested
    # if prune_model:
    #    _prune_model(model)

    return model


def pruneModelF(model, prop_var_min=0.05):
    """
    Prune a model by removing the component with the lowest variance
    if it is below a given threshold.

    Parameters
    ----------
    model : gstlearn Model object
    prop_var_min : float
        Proportion of the total variance below which a covariance
        component is suppressed.

    Returns
    -------
    bool
        True if a component has been removed, False otherwise.
    """
    ncov = model.getNCov()
    if ncov < 2:
        return False

    vartot = model.getTotalSill()
    varMin = prop_var_min * vartot
    index = None

    for icov in range(ncov):
        sill = model.getSill(icov, 0, 0)
        if sill < varMin:
            index = icov
            varMin = sill

    if index is None:
        return False

    model.delCov(index)
    return True


def _prune_model(model, prop_var_min=0.05):
    """
    Prune a model by suppressing low variance components.

    Args:
        model: gstlearn Model object
        prop_var_min: Minimum proportion of variance to keep

    Returns:
        Boolean indicating if model was pruned
    """
    ncov = model.getNCov()
    if ncov < 2:
        return False

    # Get variances
    variances = []
    for i in range(ncov):
        cov = model.getCovAniso(i)
        variances.append(cov.getSill())

    total_var = sum(variances)

    # Remove low variance components
    removed = False
    for i in range(ncov - 1, -1, -1):
        if variances[i] / total_var < prop_var_min:
            model.delCov(i)
            removed = True

    return removed and model.getNCov() > 1


def prepare_likelihood(
    db, vname, model, pol_drift=None, ext_drift=None, type="ordinary"
):
    ind = np.where(~np.isnan(db[vname]))[0]
    dbaux = gl.Db.createReduce(db, ranks=ind)
    set_var(dbaux, vname, "Var")
    ndrift = 0

    if pol_drift is None:
        pol_drift = -1 if type == "simple" else 0

    if ext_drift is not None:
        if isinstance(ext_drift, str):
            ext_drift = [ext_drift]
        set_var(dbaux, ext_drift, "Drift")
        ndrift = len(ext_drift)
    add_drifts_to_model(model, pol_drift, ndrift)

    return dbaux


def set_sill(model, sill):
    model.getCovAniso(0).setSill(sill)


def set_range(model, range):
    model.getCovAniso(0).setRangeIsotropic(range)


def model_mle(
    db,
    vname,
    pol_drift=None,
    ext_drift=None,
    struct="SPHERICAL",
    prune_proportion=0,
    aniso_model=True,
    reml=False,
    nVecchia=None,
):
    """
    Fit a gstlearn model by Maximum Likelihood.

    This function fits a covariance model to the data contained in a Db object,
    by Gaussian Maximum Likelihood. It optionally performs model pruning:
    covariance structures whose estimated variance is negligible are removed,
    and the model is refitted until no removable component remains.

    Parameters
    ----------
    db : gl.Db
        The gstlearn Db object containing coordinates and variables.

    vname : str or list[str]
        Name of the variable(s) to be fitted.

    pol_drift : int or None, optional
        Order of the polynomial drift. If None, no polynomial drift is used.

    ext_drift : str or list[str] or None, optional
        Name(s) of variables used as external drift(s).
        If None, no external drift is used.

    struct : str or list[str], optional
        List of covariance structure names (e.g. "NUGGET", "SPHERICAL").
        See `gl.printAllStruct()` for available types.

    prune_proportion : Proportions of the total variance belongs which a covariance structure is removed.
        It is performed iteratively by removing the structure with the smallest variance if it is below the
        total variance of the previously computed model. 0 means no pruning.

    aniso_model : bool, optional
        If True, allow anisotropy parameters during optimization.

    reml : bool, optional
        If True, use Restricted Maximum Likelihood (REML).

    n_vecchia : int or None, optional
        Number of neighbors for Vecchia approximation.
        If None → full likelihood is used.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'model' : gl.Model
            The fitted gstlearn model.
        - 'driftCoeffs' : numpy.ndarray
            Estimated drift coefficients (β vector).
        - 'likelihood' : float
            Value of the log-likelihood.

    Notes
    -----
    The procedure is:
    1. Build a model containing all requested structures.
    2. Fit by Maximum Likelihood (or Vecchia approximation).
    3. If `prune_model=True`, remove components with very small variance.
    4. Repeat until no structure can be removed.
    """

    # Check structure names using gstlearn
    # types = gl.checkStructNames(struct)

    ndim = db.getNDim()
    model = create_model(struct, ndim=ndim)

    dbaux = prepare_likelihood(db, vname, model, pol_drift, ext_drift)

    if nVecchia is None:
        nVecchia = -1234567

    keepgoing = True
    while keepgoing:
        mop = gl.ModelOptimParam.create(aniso_model)
        ll = gl.AModelOptimFactory.create(
            model, dbaux, None, None, None, mop=mop, nb_neighVecchia=nVecchia, reml=reml
        )
        cost = ll.run()
        if prune_proportion <= 0:
            break
        keepgoing = pruneModelF(model, prop_var_min=prune_proportion)

    return {
        "model": model,
        "driftCoeffs": model.getDriftList().getBetaHats(),
        "likelihood": -cost,
    }


def model_compute_log_likelihood(
    db, vname, model=None, pol_drift=None, ext_drift=None, reml=False, nVecchia=None
):
    """
    Fit a gstlearn model by Maximum Likelihood.

    This function fits a covariance model to the data contained in a Db object,
    by Gaussian Maximum Likelihood. It optionally performs model pruning:
    covariance structures whose estimated variance is negligible are removed,
    and the model is refitted until no removable component remains.

    Parameters
    ----------
    db : gl.Db
        The gstlearn Db object containing coordinates and variables.

    vname : str or list[str]
        Name of the variable(s) to be fitted.

    model : gstlearn Model object
        The gstlearn Model object containing the covariance structures to be used.

    pol_drift : int or None, optional
        Order of the polynomial drift. If None, no polynomial drift is used.

    ext_drift : str or list[str] or None, optional
        Name(s) of variables used as external drift(s).
        If None, no external drift is used.

    reml : bool, optional
        If True, use Restricted Maximum Likelihood (REML).

    n_vecchia : int or None, optional
        Number of neighbors for Vecchia approximation.
        If None → full likelihood is used.

    Returns
    -------
        'likelihood' : float
         Value of the log-likelihood.

    """

    # Check structure names using gstlearn
    # types = gl.checkStructNames(struct)

    ndim = db.getNDim()

    dbaux = prepare_likelihood(db, vname, model, pol_drift, ext_drift)

    if nVecchia is None:
        nVecchia = -1234567

    return model.computeLogLikelihood(dbaux)


def eval_cov_matrix(model, db):
    return model.evalCovMat(db).toTL()


def eval_drift_matrix(db, pol_drift=None, ext_drift=[]):
    model = gl.Model.createFromParam()

    if isinstance(ext_drift, str):
        ext_drift = [ext_drift]
    add_drifts_to_model(model, pol_drift, len(ext_drift), "ordinary")
    db.setLocators(ext_drift, gl.ELoc.F)
    mat = model.evalDriftMat(db).toTL()
    db.clearLocators(gl.ELoc.F)
    return mat
