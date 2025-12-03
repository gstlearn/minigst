
#' Function to create a gstlearn object containing the parameters of 
#' the neighborhood used to perform kriging
#'
#' @keywords internal
#'
.createneigh<-function(neigh){
  if (length(neigh) == 1){
    if (neigh == "unique"){
      Neigh = NeighUnique()
    }
    else if ((sum(is.numeric(neigh))== 1)){
      Neigh =NeighMoving_create(radius = neigh)
    }
  }
  else if ((sum(is.numeric(neigh))== 1) & (length(neigh) == 3)){
    if ((neigh[1] <= neigh[2]) & (neigh[3]>0)){
      Neigh =NeighMoving_create(nmini=neigh[1], nmaxi=neigh[2], radius=neigh[3])
    }
  }
  else if ((sum(is.numeric(neigh))== 1) & (length(neigh) == 2)){
    if (neigh[1] <= neigh[2]){
      Neigh =NeighMoving_create(nmini=neigh[1], nmaxi=neigh[2])
    }
  }
  else{
    stop("The neighborhood definition is not valid")
  }
  return(Neigh)
}

#' Kriging predictions and cross-validation
#'
#' Functions to compute the kriging predictions (`minikriging`) or cross-validations (`minixvalid`).
#'
#' @param dbin Db object containing the data points.
#' @param dbout Db object containing the target points (ignored for `minixvalid`), i.e. the points where the kriging predictor will be computed.
#' @param vname Name of the variable(s) to be predicted, stored as a (vector of) string(s).
#' @param model Model object containing the model used for kriging. For universal kriging, the model should be fitted on residuals.
#' @param type Type of kriging: either "simple" or "ordinary". Ignored if `polDrift` or `extDrift` is specified, in which case, universal kriging is performed.
#' @param polDrift Integer specifying the order of the polynomial drift for universal kriging. 
#' @param extDrift Name of the variable(s) specifying the external drift, stored as a (vector of) string(s), for universal kriging with external drift.
#' @param mean Mean value if simple kriging is requested. Ignored when other types of kriging are performed.
#' @param neighborhood Type of neighborhood to used, either "unique" or a vector of the form (radius), (nmin, nmax) or (nmin, nmax, radius). See details. 
#' @param std Boolean value indicating whether the kriging standard deviations (for the `minikriging` function) or standardized cross-validation errors (for the `minixvalid` function) should be computed.
#' @param prefix Prefix used to store the results
#' 
#' @details The function `minikriging` computes kriging predictions on target locations specified in a Db object (argument `dbout`) using data specified on another Db object (argument `dbin`).
#' 
#' The function `minixvalid` computes kriging 1-fold cross-validation errors on a set of points  specified in a Db object (argument `dbin`): For each point x_i, the function computes the kriging prediction at x_i when using all the other points in Db. 
#' The cross-validation error at x_i is then defined as the difference between this prediction and the actual observed value at x_i. 
#' And the standardized cross-validation error at x_i is defined as the ratio between the cross-validation error at x_i  and the kriging standard-deviation at x_i. 
#' 
#' When specifying moving neighborhoods, the vector (nmin, nmax, radius) indicates the minimum and the maximum number of points in the neighborhood and the radius of the neighborhood.
#' 
#' Finally, note that when adding external drifts, a constant drift (bias term) is automatically added as well.
#' 
#' @return The function `minikriging` directly adds the kriging predictions (and if applicable, standard-deviations) to the Db in `dbout`. The names of these newly created variables will be of the form `prefix.vname.estim` for the predictions and  `prefix.vname.stdev` for the standard deviations.
#' 
#' The function `minixvalid` directly adds the kriging cross-validation errors (and if applicable, standard-deviations) to the Db in `dbin`. The names of these newly created variables will be of the form `prefix.vname.esterr` for the cross-validation errors and  `prefix.vname.stderr` for the standardardized errors
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland") # Dataframe containing the observations
#' data("ScotlandGrid") # Dattaframe containing the target prediction locations
#'
#' # Create Observation grid
#' obsDb=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' 
#'
#' # Create directional experimental variograms along the directions 30 deg and -30 deg for the variable "Elevation"
#' varioExp = vario_exp(db=obsDb, vname="Elevation", dir=c(30,-30), nlag=20, dlag=10.)
#' # Fit a model on the resulting experimental variogram
#' struct_names = c("NUGGET","SPHERICAL", "SPHERICAL")
#' model = model_fit(varioExp, struct=struct_names, pruneModel = TRUE)
#' 
#' # Create target grid
#' targetDb=dfToDbGrid(df=ScotlandGrid,coordnames=c("Longitude","Latitude"))
#' 
#' # Perform ordinary kriging predictions and compute standard deviations for the variable `January_temp`
#' minikriging(obsDb, targetDb, vname = "January_temp", model = model, std = TRUE)
#'  
#'# Plot kriging predictions
#' dbplot_grid(targetDb,color="K.January_temp.estim")
#' dbplot_point(obsDb,size="January_temp",sizeRange=c(0.1,2),add=TRUE,title="Kriging predictions on grid")
#' 
#'# Plot kriging standard deviations
#' dbplot_grid(targetDb,color="K.January_temp.stdev")
#' dbplot_point(obsDb,size="January_temp",sizeRange=c(0.1,2),add=TRUE,title="Kriging std dev")
#' 
minikriging<-function (dbin, dbout, vname, model, type = "ordinary", polDrift = NULL, extDrift=NULL, mean = NULL, neighborhood = "unique", std = TRUE,prefix="K"){
  setVar(dbin,vname)
  Neigh = .createneigh(neighborhood)
  
  if (!is.null(extDrift) || !is.null(polDrift)){
    ## Create a copy of the model without any drift and set the mean
    mdl=Model(model)
    err = mdl$delAllDrifts()
    ## Add drifts to model
    .addDriftsToModel(mdl,polDrift,length(extDrift))
    
    ## Set drifts in databases
    setVar(dbin,extDrift,"Drift")
    setVar(dbout,extDrift,"Drift")
    
    ## Delete previous kriging result
    delVarFromDb(dbout,paste0(c(prefix,vname,"estim"),collapse = "."))
    if(std){
      delVarFromDb(dbout,paste0(c(prefix,vname,"stdev"),collapse = "."))
    }
    ## Kriging
    err = kriging(dbin=dbin, dbout=dbout, model=mdl, 
                  neigh=Neigh,
                  flag_est=TRUE, flag_std=std, flag_varz=FALSE,
                  namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `kriging` function
    Db_clearLocators(dbout,ELoc_Z())
    Db_clearLocators(dbout,ELoc_F())
    Db_clearLocators(dbin,ELoc_Z())
    Db_clearLocators(dbin,ELoc_F())
  } 
  else if (type == "simple"){
    if (!is.numeric(mean)){stop('The mean is not properly specified for the simple kriging.')}
    
    ## Create a copy of the model without any drift and set the mean
    mdl=Model(model)
    err = mdl$delAllDrifts()
    err = mdl$setMean(mean)
    
    ## Delete previous kriging result
    delVarFromDb(dbout,paste0(c(prefix,vname,"estim"),collapse = "."))
    if(std){
      delVarFromDb(dbout,paste0(c(prefix,vname,"stdev"),collapse = "."))
    }
    ## Compute kriging
    err = kriging(dbin=dbin, dbout=dbout, model=mdl, 
                  neigh=Neigh,
                  flag_est=TRUE, flag_std=std, flag_varz=FALSE,
                  namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `kriging` function
    Db_clearLocators(dbout,ELoc_Z())
    Db_clearLocators(dbin,ELoc_Z())
    
  }
  else if (type == "ordinary"){
    
    ## Create a copy of the model without any drift and set universality condition
    mdl=Model(model)
    err = mdl$delAllDrifts()
    err = mdl$addDrift(DriftM())
    
    ## Delete previous kriging result
    delVarFromDb(dbout,paste0(c(prefix,vname,"estim"),collapse = "."))
    if(std){
      delVarFromDb(dbout,paste0(c(prefix,vname,"stdev"),collapse = "."))
    }
    ## Kriging
    err = kriging(dbin=dbin, dbout=dbout, model=mdl, 
                  neigh=Neigh,
                  flag_est=TRUE, flag_std=std, flag_varz=FALSE,
                  namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `kriging` function
    Db_clearLocators(dbout,ELoc_Z())
    Db_clearLocators(dbin,ELoc_Z())
    
    } 
  else {stop("The kriging type should be either simple ordinary, or universal (with a drift properly defined)")}
  
  return(invisible(NULL))
}


#' #' Compute the cross validation error
#' #'
#' #' Function to perform the (1-fold) cross validation for a given kriging exercise.
#' #'
#' #' @param dbin Db containing the data.
#' #' @param vname Name of the variable, stored as a string.
#' #' @param model A \pkg{gstlearn} Model object containing a fitted model.
#' #' @param type Type of kriging, either "simple" or "ordinary".
#' #' @param mean Mean value if simple kriging is requested 
#' #' @param neighborhood Type of neighborhood to used, either "unique" or a vector of the form (radius), (nmin, nmax) or (nmin, nmax, radius) see details. 
#' #' @param std Boolean value indicating whether the prediction standard deviation should be computed.
#' #'
#' #' @details For each observation location x_i, compute the kriging prediction at x_i when using all the observations except the one located at x_i. 
#' #' The cross-validation error at x_i is then defined as the difference between this prediction and the actual observed value at x_i.
#' #' 
#' #' @return a db containing the same variables as the input db + two new variables: 
#' #' One variable (with name ending with .esterr) containing the cross-validation errors, 
#' #' and one variable (with name ending with .stderr) containing the cross-validation errors normalized by the kriging standard deviation.
#' #'
#' #' @export
#' #'
#' #' @examples
#' #' library(minigst)
#' #'
#' #' #todo
#' #' 
#' #' 
#' #'  
#' #'
#' xvalidation <- function (dbin, vname, drift = NULL, model, type = "ordinary",prefix = "OK", mean = NULL, neighborhood = "unique", std = TRUE){
#'   setVar(dbin,vname)
#'   Neigh = .createneigh(neighborhood)
#'   if (type == "simple"){
#'     if (!as.numeric(mean)){stop('The mean is not properly specified for the simple kriging.')}
#'     err = model$setMean(mean)
#'     err = xvalid(db=dbin, model=model, 
#'                  neigh=Neigh,
#'                  flag_xvalid_est=1, flag_xvalid_std=1,  
#'                  namconv=NamingConvention_create(paste0("Xvalid_",prefix), flag_locator = FALSE)
#'     )
#'   }
#'   else if (type == "ordinary"){
#'     err = model$delAllDrifts()
#'     err = model$addDrift(DriftM())
#'     err = xvalid(db=dbin, model=model, 
#'                  neigh=Neigh,
#'                  flag_xvalid_est=1, flag_xvalid_std=1,  
#'                  namconv=NamingConvention_create(paste0("Xvalid_",prefix), flag_locator = FALSE)
#'     )
#'   }
#'   else if (type == "universal" & !is.null(drift)){
#'     for (xvar in drift){
#'       dbin$setLocator(xvar,ELoc_F())
#'     }
#'     err = xvalid(db=dbin, model=model, 
#'                  neigh=Neigh,
#'                  flag_xvalid_est=1, flag_xvalid_std=1,  
#'                  namconv=NamingConvention_create(paste0("Xvalid_",prefix), flag_locator = FALSE)
#'     )
#'   }
#'   else {stop("The kriging type should be either simple, ordinary, or universal")}
#'   return(dbin)
#' }



#' @rdname minikriging
#'
#' @export
#'
minixvalid<-function (dbin, vname, model, type = "ordinary", polDrift = NULL, extDrift=NULL, mean = NULL, neighborhood = "unique", std = TRUE,prefix="Xvalid"){
  setVar(dbin,vname)
  Neigh = .createneigh(neighborhood)
  
  if (!is.null(extDrift) || !is.null(polDrift)){
    ## Create a copy of the model without any drift and set the mean
    mdl=Model(model)
    err = mdl$delAllDrifts()
    ## Add drifts to model
    .addDriftsToModel(mdl,polDrift,length(extDrift))
    
    ## Set drifts in databases
    setVar(dbin,extDrift,"Drift")
    
    ## Delete previous xvalid result
    delVarFromDb(dbin,paste0(c(prefix,vname,"esterr"),collapse = "."))
    if(std){
      delVarFromDb(dbin,paste0(c(prefix,vname,"stderr"),collapse = "."))
    }
    ## xvalid
    err = xvalid(dbin, model=mdl, 
                 neigh=Neigh,
                 flag_xvalid_est=TRUE, flag_xvalid_std=std,
                 namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `xvalid` function
    Db_clearLocators(dbin,ELoc_Z())
    Db_clearLocators(dbin,ELoc_F())
  }
  else if (type == "simple"){
    if (!is.numeric(mean)){stop('The mean is not properly specified for the simple xvalid.')}
    
    ## Create a copy of the model without any drift and set the mean
    mdl=Model(model)
    err = mdl$delAllDrifts()
    err = mdl$setMean(mean)
    
    ## Delete previous xvalid result
    delVarFromDb(dbin,paste0(c(prefix,vname,"esterr"),collapse = "."))
    if(std){
      delVarFromDb(dbin,paste0(c(prefix,vname,"stderr"),collapse = "."))
    }
    ## Compute xvalid
    err = xvalid(dbin, model=mdl, 
                  neigh=Neigh,
                  flag_xvalid_est=TRUE, flag_xvalid_std=std, 
                  namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `xvalid` function
    Db_clearLocators(dbin,ELoc_Z())
    
  }
  else if (type == "ordinary"){
    
    ## Create a copy of the model without any drift and set universality condition
    mdl=Model(model)
    err = mdl$delAllDrifts()
    err = mdl$addDrift(DriftM())
    
    ## Delete previous xvalid result
    delVarFromDb(dbin,paste0(c(prefix,vname,"esterr"),collapse = "."))
    if(std){
      delVarFromDb(dbin,paste0(c(prefix,vname,"stderr"),collapse = "."))
    }
    ## xvalid
    err = xvalid(dbin, model=mdl, 
                  neigh=Neigh,
                  flag_xvalid_est=TRUE, flag_xvalid_std=std,
                  namconv=NamingConvention(prefix)
    )
    ## Remove locators automatically assigned by the `xvalid` function
    Db_clearLocators(dbin,ELoc_Z())
    
  }
  else {stop("The xvalid type should be either simple ordinary, or universal (with a drift properly defined)")}
  
  return(invisible(NULL))
}

