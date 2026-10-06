
#' Compute the Kriging mean of a variable
#'
#' Function that calculates the kriging mean of a variable in a database.
#'
#' @param db Db object containing the observations.
#' @param vname Name of the variable of interest.
#' @param model Model object giving the covariance/variogram model of the data.
#'
#' @return The function returns a numeric value giving the kriging mean of the variable of interest.
#'
#' @export
#'
#' @examples
#' library(minigst)
#' 
#' # Database
#' data("Scotland")
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#'
#' # Model
#' varioExp = vario_exp(db=db, vname="Elevation", dir=c(30,-30), nlag=20, dlag=10.)
#' model = model_fit(varioExp, struct=c("NUGGET","SPHERICAL", "SPHERICAL"), pruneModel = TRUE)
#' 
#' # Kriging mean
#' km = kriging_mean(db = db, vname = "Elevation", model = model)
#' km
#' 
kriging_mean <- function(db, vname, model){
  
  setVar(db,vname)
  addSel(db,!is.na(db[vname]))
  tgt = createDbGrid(coords=as.list(db$getCenters(useSel=TRUE)),coordnames=c("xcoord", "ycoord")) 
  neighU = NeighUnique_create()
  mod = model$clone()
  
  # Simple Kriging of target with known mean 0
  err = kriging(dbin=db, dbout=tgt, model = mod, neigh=neighU, flag_est=TRUE, flag_std=FALSE, flag_varz=FALSE, namconv=NamingConvention("SK0"))
  # Simple Kriging of target with known mean 1
  err = mod$setMean(mean=1)
  err = kriging(dbin=db, dbout=tgt, model = mod, neigh=neighU, flag_est=TRUE, flag_std=FALSE, flag_varz=FALSE, namconv=NamingConvention("SK1"))
  # Ordinary Kriging of target
  err = mod$addDrift(DriftM())
  err = kriging(dbin=db, dbout=tgt, model = mod, neigh=neighU, flag_est=TRUE, flag_std=FALSE, flag_varz=FALSE, namconv=NamingConvention("OK"))
    
  # Kriging mean
  km = (tgt["OK*"]-tgt["SK0*"])/(tgt["SK1*"]-tgt["SK0*"])
  
  return(km)
  
}

#' Set the mean in a model
#'
#' Function that sets the value of the mean in a Model object.
#'
#' @param model Model object.
#' @param mu Value of the mean.

#' @details ...
#'
#' @return The function updates \code{model} and returns nothing.
#'
#' @export
#'
#' @examples
#' library(minigst)
#' 
#' # Database
#' data("Scotland")
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#'
#' # Model
#' varioExp = vario_exp(db=db, vname="Elevation", dir=c(30,-30), nlag=20, dlag=10.)
#' model = model_fit(varioExp, struct=c("NUGGET","SPHERICAL", "SPHERICAL"), pruneModel = TRUE)
#' 
#' # Kriging mean
#' km = kriging_mean(db = db, vname = "Elevation", model = model)
#' 
#' # Set the mean in the model
#' setMean(model,km)
#' 
setMean <- function(model,mu){
  err = model$setMeans(mu)
  return(invisible(NULL))
}


#' Simulate a Gaussian random field with the turning bands algorithm
#'
#' Function that adds simulations of a Gaussian random field to a database that indicates the target locations.
#'
#' @param target Db object containing the target points on which we want to simulate the model, and to which the result will be added.
#' @param model Model object defining the model we want to simulate.
#' @param nsim Number of samples to generate.
#' @param ntuba Number of turning bands to use (Default 100).
#' @param seed Optional integer specifying the seed of the algorithm for reproducibility purposes (defaults to 43431).
#' @param simname Optional name of the variable displaying the simulated values in the database.
#' @param dbcond Optional Db object containing the conditioning data.
#' @param vcond Optional character specifying which variable in \code{dbcond} is to be used as conditioning data.

#' @details ...
#'
#' @return The function updates \code{target} and returns nothing.
#'
#' @export
#'
#' @examples
#' library(minigst)
#' 
#' # Target grid
#' tgt=createDbGrid(coords=list(seq(from=0,to=1,length.out=100),seq(from=0,to=1,length.out=100)),coordnames=c("xcoord", "ycoord"))
#' 
#' # Model
#' mod=createModel(struct=c("EXPONENTIAL", "NUGGET"), range = 0.3, sill = c(1,0.1), ndim=2, mean = 3)
#'
#' # Simulation
#' minirgrf(target=tgt, model=mod, nsim = 1, simname = "Mysim") 
#' dbplot_grid(db=tgt,color="Mysim", cmap = "Spectral", title = "Simulation", colorLegendTitle= "Value")
#' 
minirgrf <- function(target, model, nsim, ntuba = 1000, seed = 43431, simname = NULL, dbcond = NULL, vcond = NULL){
  
  if(is.null(dbcond)){ # Unconditional simulations
    if (!is.null(simname)){
      if(nsim==1){
        sim_names=simname
      }else{
        sim_names=paste0(simname,".S",1:nsim)
      }
      delVarFromDb(target,sim_names)
      err = simtub(dbout=target, model=model, nbsimu=nsim, nbtuba=ntuba, seed=seed, namconv=NamingConvention(simname))
    }else{
      if(nsim==1){
        sim_names="Simu"
      }else{
        sim_names=paste0("Simu",".S",1:nsim)
      }
      delVarFromDb(target,sim_names)
      err = simtub(dbout=target, model=model, nbsimu=nsim, nbtuba=ntuba, seed = seed, namconv=NamingConvention("Simu"))
    }

    ## Remove locators automatically assigned by the `minigrf` function
    Db_clearLocators(target,ELoc_Z())
    
  }else{ # Conditional simulations
    
    if(length(intersect(colnames(dbcond[]),vcond))==0){
      stop("Check the variable names: one or several of the supplied names are absent from the Db.")
    }
    setVar(dbcond,vcond)
    neighU = NeighUnique_create()
    defineDefaultSpace(ESpaceType_RN(), ndim = target$getNDim())
    
    if (!is.null(simname)){
      if(nsim==1){
        sim_names=paste0(simname,".",vcond)
      }else{
        sim_names=paste0(simname,".",vcond,".S",1:nsim)
      }
      delVarFromDb(target,sim_names)
      err = simtub(dbin = dbcond, dbout=target, model=model, nbsimu=nsim, nbtuba=ntuba, neigh = neighU, seed=seed, namconv=NamingConvention(simname))
    }else{
      if(nsim==1){
        sim_names=paste0("Simu",".",vcond)
      }else{
        sim_names=paste0("Simu",".",vcond,".S",1:nsim)
      }
      delVarFromDb(target,sim_names)
      err = simtub(dbin = dbcond, dbout=target, model=model, nbsimu=nsim, nbtuba=ntuba, neigh = neighU, seed = seed)
    }
    
    ## Remove locators automatically assigned by the `minigrf` function
    Db_clearLocators(dbcond,ELoc_Z())
    Db_clearLocators(target,ELoc_Z())
  }

  return(invisible(NULL))
}
