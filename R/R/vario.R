
#' Function to adapt the size of the model parameters passed to gstlearn functions
#' If n>0, it makes sure the argument arg has size 1 or n, and returns a copy of arg with size n.
#'
#' @keywords internal
#'
.rep_arg<-function(n,arg,argname){
  
  if(length(arg)==1){
    return(rep(arg,max(1,n)))
  }else if(n==0){
    stop(paste0("The length of the argument ",argname," should be ",1,"."))
  }else if(length(arg)!=n){
    stop(paste0("The length of the argument ",argname," should be ", ifelse(n>1,paste0("either ",1," or ",n,"."),"1.")))
  }else{
    return(arg)
  }
}

#' Function to create a gstlearn object containing the parameters needed to compute
#' an experimental variogram.
#' See the documentation of the function \code{vario_exp}
#' for details about the parameters.
#'
#' @keywords internal
#'
.createVarioParam<-function(db,dir=NULL,nlag=20, dlag=100,
                            toldis = 0.5, tolang= NULL){
  angles=F
  if(is.null(dim(dir))){
    if(db$getNDim()>2){
      stop("The directions have to be specified by vectors when the space dimension of the Db (",db$getNDim(),")is greater than 2.")
    }
    ndir=length(dir)
    angles=T
  }else{
    ndir=nrow(dir)
    ndim=ncol(dir)
    if(ndim!=db$getNDim()){
      stop("The number of columns of dir (",ndim,") should be the same as the space dimension in the Db (",db$getNDim(),").")
    }
  }
  
  nlag=.rep_arg(ndir,nlag,"nlag")
  dlag=.rep_arg(ndir,dlag,"dlag")
  toldis=.rep_arg(ndir,toldis,"toldis")
  if((is.null(tolang)) || (ndir==0)){
    tolang=rep(180/(2*ndir),max(1,ndir))
  }else{
    tolang=.rep_arg(ndir,tolang,"tolang")
  }
  
  
  if(ndir>0){
    varioParam=VarioParam()
    for(i in 1:ndir){
      if(angles){
        direc=DirParam_create(angle2D = dir[i],nlag=nlag[i], dlag=dlag[i], toldis = toldis[i],tolang = tolang[i])
      }else{
        direc=DirParam_create(codir = dir[i,],nlag=nlag[i], dlag=dlag[i], toldis = toldis[i],tolang = tolang[i])
      }
      varioParam$addDir(direc)
    }
    
  }else{
    varioParam = VarioParam_createOmniDirection(nlag=nlag, dlag=dlag, toldis = toldis)
  }
  
  return(varioParam)
}


#' Compute an experimental variogram
#'
#' Function to compute an experimental variogram for a single variable, or experimental variogram and cross-variograms for a set of variables.
#'
#' @param db Db object.
#' @param vname Name(s) of the variable(s), stored as a (vector of) string(s).
#' @param polDrift Integer specifying the order of the polynomial drift for universal kriging. 
#' @param extDrift Name of the variable(s) specifying the external drift, stored as a (vector of) string(s), for universal kriging with external drift.
#' @param dir Direction used to compute the variogram. The default (\code{NULL}) reverts to an omnidirectional variogram. The directions can be specified either by a vector of angles in degrees (only in 2D and the value(s) denote the angle between the horizontal axis and the desired direction), or as a matrix whose rows contain direction vectors.
#' @param nlag Number of lags used to compute the variograms.
#' @param dlag Distance of the lags.
#' @param toldis Tolerance on the distance (between 0 and 1). See \emph{Details}.
#' @param tolang Tolerance on the angles (between 0 and 90). See \emph{Details}.
#'
#' @details The parameter \code{toldis} represents a tolerance on the distance when computing the variogram value, and is expressed as proportion.
#' For  a given lag value \eqn{h}, all the pairs separated by distance between (\code{h}-\code{toldis}*\code{dlag}) and (\code{h}+\code{toldis}*\code{dlag})
#' are used to compute the corresponding variogram value \eqn{gamma(h)}.
#'
#' The parameter \code{tolang} represents a tolerance on the angle and is expressed as an angle.
#' For a given lag value \eqn{h} and a given angle \eqn{theta}, all the pairs with an angle between (\eqn{theta}-\code{tolang}) and (\eqn{theta}+\code{tolang})
#' are used to compute the corresponding variogram value \eqn{gamma(h)}.
#' 
#' Finally, note that when adding external drifts, a constant drift (bias term) is automatically added as well.
#'
#' @return The function returns a \pkg{gstlearn} object containing the experimental variogram(s) (and cross-variograms).
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland")
#'
#' # Create Db
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db$display()
#'
#' # Create directional experimental variograms along the directions 30 deg and -30 deg for the variable "Elevation".
#' varioExpDir = vario_exp(db=db, vname="Elevation", dir=c(30,-30), nlag=20, dlag=10.)
#' # varioExpDir$display() # Run to display the content of the experimental variogram object.
#' # Plot experimental variogram
#' plot_vario(varioExpDir, pairDisplay="size", title="Experimental variograms for the variable 'Elevation'")
#'
#' # Create directional experimental (cross-)variograms along the directions 30 deg and -30 deg for the variables "Elevation" and "January_temp".
#' varioExpDir2var = vario_exp(db=db, vname=c("Elevation","January_temp"), dir=c(30,-30), nlag=20, dlag=10.)
#' # varioExpDir2var$display() # Run to display the content of the experimental variogram object.
#' # Plot experimental variogram
#' dev.new() # open new window
#' plot_vario(varioExpDir2var, title="Experimental (cross)-variograms for the variables 'Elevation' and 'January_temp")
#'
#'
vario_exp<-function(db,vname, polDrift = NULL, extDrift=NULL,dir=NULL,nlag=20, dlag=100,
                    toldis = 0.5, tolang= 22.5){
  setVar(db,vname)
  
  varioParam= .createVarioParam(db,dir,nlag, dlag,toldis,tolang)
  
  varioexp = Vario(varioParam)

  if (!is.null(extDrift) || !is.null(polDrift)){
    if (!is.null(extDrift))
    {
      setVar(db,extDrift,"Drift")
    }
    EDmodel = Model_create()
    ## Add drifts to model
    .addDriftsToModel(EDmodel,polDrift,length(extDrift))
    
    err = varioexp$compute(db,model=EDmodel)
  }else{
    err = varioexp$compute(db)
  }
  
  return(varioexp)
}




#' Compute a variogram cloud
#'
#' Function to compute a variogram cloud for a variable in a `gstlearn` Db object.
#'
#' @param db Db object.
#' @param vname Name of the variable.
#' @param dir Direction used to compute the variogram. Can be either \code{NULL} (Default) for an omnidirectional variogram cloud, or a single direction specified either by an angle in degrees (only in 2D and the value denote the angle between the horizontal axis and the desired direction), or as a matrix with a single row containing a direction vector.
#' @param gridRes Number or vector of size 2 specifying the resolution of the computation grid. See \emph{Details}.
#' @param tolang Tolerance on the angles. See \emph{Details}.
#'
#' @details The variogram cloud is computed as a 2D grid with dimensions given by \code{gridRes}. The first axis discretizes the set of possible distances between pairs of points in \code{db}.
#' The second axis discretizes the set of possible variogram values between between pairs of points in \code{db}. The value computed at a given grid node corresponds to the number of pairs of points in \code{db} falling in the corresponding grid cell.
#'
#' @return A DbGrid containing the computed variogram cloud, where the number of pairs are stored in each grid cell are stored in a variable called `Cloud.vname`.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland")
#'
#' # Create Db
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db$display()
#'
#' # Compute and plot the omnidirectional variogram cloud of the variable "Elevation".
#' varioCloudGrid = vario_cloud(db=db, vname="Elevation")
#' dbgrid_plot(varioCloudGrid,color="Cloud.Elevation",legendTitle="Nb of pairs")
#'
vario_cloud<-function(db,vname,dir=NULL, gridRes=100, tolang= 22.5){
  plot=FALSE
  
  if(length(vname)!=1){
    stop("Variogram clouds are only computed for a single variable: please specify 1 variable name for the argument vname.")
  }
  setVar(db,vname)
  
  ndir=0
  if(!is.null(dim(dir))){
    ndir=nrow(dir)
  }else{
    ndir=length(dir)
  }
  if(ndir>1){
    stop("Variogram clouds are only computed for a single direction: please specify only 1 direction or use NULL to create an omnidirectional variogram cloud.")
  }
  
  stopifnot(is.numeric(gridRes))
  if(length(gridRes)==1){
    gridRes=c(gridRes, gridRes)
  }
  
  varioParam= .createVarioParam(db,dir=dir,tolang=tolang)
  
  grid.cloud=db_vcloud(db, varioParam,lagnb = gridRes[1], varnb = gridRes[2])
  if(plot){ #<- Bug quand plot=TRUE
    vn=grid.cloud$getNames("Cloud.*")[1]
    p=dbplot_grid(grid.cloud,color=vn,legendTitle =  "Nb of pairs",cmap='RdBu',
                  xlab = "Distance", ylab = "Variogram", title = paste0("Variogram cloud: ",vname))
    print(p)
  }
  return(grid.cloud)
}



#' Compute a variogram map
#'
#' Function to compute a variogram map for a variable in a `gstlearn` Db object.
#'
#' @param db Db object.
#' @param vname Name of the variable.
#' @param gridRes Number specifying the resolution of the computation grid. See \emph{Details}.
#'
#' @details The variogram map is computed on a 2D grid, centered at the origin, and with dimensions (2*\code{gridSize}+1) * (2*\code{gridSize}+1).
#'
#' @return A DbGrid containing the computed variogram map.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland")
#'
#' # Create Db
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db$display()
#'
#' # Compute and plot the omnidirectional variogram cloud of the variable "Elevation".
#' varioMapGrid = vario_map(db=db, vname="Elevation", plot = TRUE)
#'
vario_map<-function(db,vname,gridRes=20){
  plot=F
  setVar(db,vname)
  
  stopifnot(is.numeric(gridRes))
  grid.vmap = db_vmap(db, nxx = rep(gridRes[1],db$getNDim()))
  
  if(plot){
    vn1=grid.vmap$getNames("VMAP.*.Var")[1]
    p1=dbplot_grid(grid.vmap,color= vn1,cmap="Spectral",
                   legendTitle ="Var",
                   title = paste0("Variogram map : ",vname))
    
    vn2=grid.vmap$getNames("VMAP.*.Nb")[1]
    p2=dbplot_grid(grid.vmap,color=vn2,cmap="RdBu",
                   legendTitle ="Nb",
                   title = "Number of pairs")

    print(ggarrange(p1,p2,nrow=1,ncol=2, common.legend = FALSE))
  }

  
  return(grid.vmap)
}




#' Fit a model on experimental variograms or maps
#'
#' Function to fit a model on an experimental variogram, a set of experimental (cross-)variograms or a variogram map.
#'
#' @param vario Experimental variogram(s) or variogram map (as a \pkg{gstlearn} object). See \emph{Details}.
#' @param polDrift Integer specifying the order of the polynomial drift. 
#' @param extDrift Name of the variable(s) specifying the external drift, stored as a (vector of) string(s).
#' @param struct Vector containing the names of the desired basic structures. The list of available structures is obtained by calling the function \code{printAllStruct()}.
#' @param pruneModel Whether or not to prune the model. See \emph{Details}.
#' @param anisoModel Whether or not to fit an anistropic model.
#'
#' @details The argument \code{vario} is specified as the output of the \code{vario_exp} function (for experimental variograms)
#' or as the output of the \code{vario_map} function (for variogram maps).
#'
#' The fitting function first tries to fit a model containing all the structures specified in \code{struct}. if \code{pruneModel = TRUE},
#' this model is pruned, i.e. the structures associated with negligible variances are discarded from the model, and a new fit is performed.
#' These last two steps are repeated until no basic structures can be removed.
#' 
#' Note that fitting variogram maps using drifts is not supported yet.
#'
#' @return A \pkg{gstlearn} Model object containing the fitted model.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland")
#'
#' # Create Db
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db$display()
#'
#' # Create directional experimental variograms along the directions 30 deg and -30 deg for the variable "Elevation"
#' varioExp = vario_exp(db=db, vname="Elevation", dir=c(30,-30), nlag=20, dlag=10.)
#'
#' # Fit a model on the resulting experimental variogram
#' struct_names = c("NUGGET","SPHERICAL", "SPHERICAL")
#' model = model_fit(varioExp, struct=struct_names, pruneModel = TRUE)
#' model$display() # Display the content of the model
#'
#' # Plot the experimental variogram and the fitted model
#' plot_vario(varioExp,model = model, pairDisplay = "size",
#'            title="Model adjustment for Elevation")
#'
#'
model_fit<-function(vario,polDrift=NULL,extDrift=NULL,struct="SPHERICAL",pruneModel=TRUE,anisoModel=TRUE){
  types = .checkStructNames(struct)
  model = Model()
  if(class(vario)=="_p_gstlrn__Vario"){
    
    if (!is.null(extDrift) || !is.null(polDrift)){
      ## Add drifts to model
      .addDriftsToModel(model,polDrift,length(extDrift))
    } 
    err = model$fit(vario, types=types, optvar=Option_VarioFit(flag_noreduce=pruneModel,auth_aniso=anisoModel))
  
  }else if(class(vario)=="_p_gstlrn__DbGrid"){
    vn=vario$getNames("VMAP.*.Var")[1]
    if(is.na(vn)){
      stop("The DbGrid supplied in 'vario' should contain a variogram map saved as a variable whose name is of the form 'VMAP.*.Var'.")
    }
    setVar(vario,vn)
    err = model$fitFromVMap(vario, types=types, optvar=Option_VarioFit(flag_noreduce=pruneModel,auth_aniso=anisoModel))
  }else{
    stop("The argument 'vario' expects either an experimental variogram or a DbGrid containing a variogram map.")
  }
  
  return(model)
}


#' Basic structures for models
#'
#' Get the list of all available basic structures/covariance functions in \pkg{gstlearn}.
#'
#'
#' @return `getAllStruct` returns a dataframe (with two columns `Name` and `Description`) listing all the available basic structures, whereas `printAllStruct` only prints this dataframe and returns nothing.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Print list of basic structures
#' printAllStruct()
#'
#' # Store the name of the 3rd covariance structure in the list ("SPHERICAL")
#' struct_name=getAllStruct()$Name[3]
#' print(struct_name)
#' 
getAllStruct<-function(){
  ## Available structures
  vnames=ECov_getAllKeys()
  ## Remove "weird" choices
  id_sel=which(!(vnames %in% c("UNKNOWN","FUNCTION")))
  
  AllStruct=data.frame("Name"=ECov_getAllKeys()[id_sel],"Description"=ECov_getAllDescr()[id_sel])
  return(AllStruct)
}

#' @rdname getAllStruct
#'
#' @export
#'
printAllStruct<-function(){
  print(getAllStruct())
  return(invisible(NULL))
}


#'
#' Wrapper for the function that doesn't crash R if names are mispecified
#'
#' @keywords internal
#'
.checkStructNames<-function(struct_names){
  AllStruct=getAllStruct()
  for(str in struct_names){
    # Search in list of available structure names (search names converted to lowercase to make the search case-insensitive)
    is_found=tolower(str) %in% tolower(AllStruct$Name)
    if(!is_found){
      stop("Structure name '",str,"' is not part of the available names. Please choose a name from: ", paste(AllStruct$Name,collapse=", "))
    }
  }
  return(ECov_fromKeys(struct_names))
}

#'
#'Function to check the covariance parameters proposed by the user
#' @keywords internal
#'
.checkCovParam<-function(param,nameParam,n){
  paramf=param
  if((!is.numeric(param)) || (sum(is.na(param)>0))){
    stop("The argument ",nameParam,"should be numeric, and NA are not allowed.")
  }
  if(length(param)==1){
    paramf=rep(param,n)
  }else if(length(param)!=n){
    stop(paste0("The length of ", nameParam," (",length(param),") must be the same as the number of structures (",n,")"))
  }
  return(paramf)
}


#' Create a model
#'
#' Function to create a \pkg{gstlearn} Model object (corresponding to a single variable with an isotropic covariance function).
#'
#' @param struct Vector containing the names of the desired basic structures. The list of available structures is obtained by calling the function \code{\link{printAllStruct}}.
#' @param range Value or vector specifying the range (see `isScale` for interpretation) of each structure. Must be of size 1 (if all the structures share the same value) or have the same size as \code{struct}.
#' @param sill Value or vector specifying the sill/variance of each structure. Must be of size 1 (if all the structures share the same value) or have the same size as \code{struct}.
#' @param param Value or vector specifying the extra parameter (eg. smoothness parameter for Matérn covariances) of each structure. Must be of size 1 (if all the structures share the same value) or have the same size as \code{struct}.
#' @param ndim Space dimension of the model.
#' @param mean Mean of the model (0 by default).
#' @param isScale Whether the specified `range` correspond to a correlation length (`FALSE`, choice by default) or a scaling factor (`TRUE`).
#' 
#' @return A \pkg{gstlearn} Model object.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' ## Parameters of the model
#' struct_names = c("SPHERICAL", "K-BESSEL")
#' ranges=c(0.3,0.5)
#' variances=c(0.1,1)
#' params=c(1,1) # The first value will be ignored since the corresponding covariance function ("SPHERICAL") does not require an extra parameter.
#'
#' ## Create model
#' model=createModel(struct=struct_names, range = ranges, sill = variances, param = params,ndim=2)
#'
createModel<-function(struct="SPHERICAL", range = 0.3, sill = 1, param = 1,ndim=2, mean=0,isScale=FALSE){
  
  nstruct=length(struct)
  range=.checkCovParam(range,"range",nstruct)
  sill=.checkCovParam(sill,"sill",nstruct)
  param=.checkCovParam(param,"param",param)
  
  ## Create empty model for 1 variable, and spaceDim dimension
  model = Model(1,ndim)
  for(i in 1:nstruct){
    err=Model_addCovFromParam(model,type = .checkStructNames(struct[i])[1][[1]], sill = sill[i], range=range[i],param=param[i],flagRange = !isScale) ## Ajouter un nugget de variance 1
  }
  model$setMeans(mean)
  
  return(model)
  
}



#' Function to evaluate a model between two points.
#' See \code{model_eval} for details on the parameters.
#'
#' @keywords internal
#'
.model_eval_base<-function(x,y,model,mode){
  if(mode=="VG"){
    md=CovCalcMode(); md$setAsVario(TRUE);
  }else if(mode=="COV"){
    md=CovCalcMode(); md$setAsVario(FALSE);
  }else{
    stop("Only values are possible for the argument mode: 'VG' to compute variogram values, and 'COV' to compute covariance values.")
  }
  ndim=model$getNDim()
  if(!((length(x)==ndim)&&(length(y)==ndim))){
    stop("Wrong dimensions for coordinate vectors when evaluating the model.")
  }
  return(Model_eval(model,SpacePoint(x),SpacePoint(y),mode = md))
}



#' Evaluate a model
#'
#' Function to evaluate a \pkg{gstlearn} Model object, i.e. to compute the value of the associated covariance function on a set of points.
#'
#' @param x,y Matrices (of same size) whose rows specify the pairs of points at which the covariance is evaluated. The number of columns of these matrices must be the same as the space dimension of the model.
#' @param h Vector of lag values at which the covariance is evaluated. These lags are taken in the direction specified by \code{dir}.
#' @param dir Direction vector at which the lags in \code{h} are taken.  The size of \code{dir} must be the same as the space dimension of the model.
#' @param model Model object to evaluate.
#' @param mode Whether to compute the variogram values (\code{mode="VG"}) or covariance values (\code{mode="VG"}).
#'
#' @details There are two ways of specifying where the model is valuated:
#' \itemize{
#'   \item Either by specifying both the arguments \code{x} and \code{y}.
#'   \item Or by specifying the argument \code{h} and optionally the argument \code{dir} (whose default value \code{NULL} means that the direction along the vector of ones is taken).
#' }
#'
#'
#' @return A vector containing the evaluations of the model at the specified targets.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' ## Create model
#' model=createModel(struct=c("EXPONENTIAL", "NUGGET"), range = 0.3, sill = c(1,0.1), ndim=2)
#'
#' ## Evaluate the model along the direction (0,1)
#' hseq=seq(from=0,to=1,length.out=100)
#' md_eval_lags=model_eval(h=hseq,dir=c(0,1), model=model,mode="COV")
#'
#' ## Plot Variogram along the direction (0,1) using the function curve
#' curve(model_eval(h=x, dir=c(0,1), model=model,mode="VG"),from = 0, to =1,n = 10^3,xlab="Lag",ylab="Variogram")
#'
#' ## Evaluate the model on some random points in the unit square
#' xx=matrix(runif(20),nrow=10,ncol=2)
#' yy=matrix(runif(20),nrow=10,ncol=2)
#' md_eval_pts=model_eval(x=xx,y=yy,model=model,mode="COV")
#'
#'
#'
model_eval<-function(x=NULL,y=NULL,h=NULL,dir=NULL, model=createModel(),mode="COV"){
  ndim=model$getNDim()
  res=NULL
  if((!is.null(x))&&(!is.null(y))){
    if(!((ncol(x)==ndim)&&(ncol(y)==ndim)&&(nrow(x)==nrow(y)))){
      stop(paste0("Wrong dimensions for coordinate matrice x or y when evaluating the model. The number of rows and columns of x and y must be the same, and the number of columns must be equal to the space dimension of the model (",ndim,")."))
    }else{
      res=sapply(1:nrow(x),function(i){.model_eval_base(x[i,],y[i,],model,mode)})
    }
  }else if(!is.null(h)){
    if(is.null(dir)){
      dir=rep(1,ndim)/sqrt(ndim)
    }else{
      if(length(dir)!=ndim){
        stop(paste0("Wrong dimensions for direction vector dir when evaluating the model. The length of dir must be equal to the space dimension of the model (",ndim,")."))
      }
    }
    res=sapply(h,function(hh){.model_eval_base(rep(0,ndim),hh*dir,model,mode)})
  }else{
    stop("You must specify either the argument h, or both the arguments x and y.")
  }
  
  return(res)
}





#' Compute a covariance or variogram matrix from a model
#'
#' Function to compute a covariance or variogram matrix between pairs of points, from a \pkg{gstlearn} Model object.
#'
#' @param x,y Matrices (of same size) whose rows specify the pairs of points at which the covariance is evaluated. The number of columns of these matrices must be the same as the space dimension of the model.
#' @param model Model object to evaluate.
#' @param mode Whether to compute the variogram values (\code{mode="VG"}) or covariance values (\code{mode="VG"}).
#'
#' @details If the argument \code{y} is set to \code{NULL} (Default), then \code{y} is set to \code{x}.
#'
#' @return A matrix of size \code{nrow(x)}*\code{nrow(y)} containing the covariance/variogram between each point in \code{x} and each point in \code{y}.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' ## Create model
#' model=createModel(struct=c("EXPONENTIAL", "NUGGET"), range = 0.3, sill = c(1,0.1), ndim=2)
#'
#' ## Evaluate the covariance matrix on some random points in the unit square
#' xx=matrix(runif(20),nrow=10,ncol=2)
#' yy=matrix(runif(10),nrow=5,ncol=2)
#' covMat=model_covMat(x=xx,y=yy,model=model,mode="COV")
#'
#'
#'
model_covMat<-function(x,y=NULL,model=createModel(),mode="COV"){
  
  ndim=model$getNDim()
  if(ncol(x)<ndim){
    stop(paste0("The number of columns of x should be the same as the space dimension of model (",ndim,")"))
  }
  
  if(mode=="VG"){
    md=CovCalcMode(); md$setAsVario(TRUE);
  }else if(mode=="COV"){
    md=CovCalcMode(); md$setAsVario(FALSE);
  }else{
    stop("Only values are possible for the argument mode: 'VG' to compute variogram values, and 'COV' to compute covariance values.")
  }
  
  x=as.data.frame(x)
  db1=dfToDb(x,colnames(x)[1:ndim])
  
  if(!is.null(y)){
    if(ncol(y)<ndim){
      stop(paste0("The number of columns of y should be the same as the space dimension of model (",ndim,")"))
    }
    y=as.data.frame(y)
    db2=dfToDb(y,colnames(y)[1:ndim])
  }else{
    db2=NULL
  }

  if(is.null(db2)){
    res=Model_evalCovMatSym(model,db1,mode=md)$toTL()
    # res=Model_evalCovMatrix(model,db1,mode=md)$toTL()
  }else{
    res=Model_evalCovMatrix(model,db1,db2,mode=md)$toTL()
  }
  
  return(res)
}







#' Extract parameters from a Model object
#'
#' Functions to extract the parameters of a Model Object
#'
#' @param model Model object
#'
#' @details Let `Nstruct` be the number of basic structures in the model, and `Nvar` its number of  variables, and `Ndim` the number of space dimsensions.
#' \itemize{
#'   \item \code{model_getStructNames} returns a vector of size `Nsctruct` containing the names of the basic structures composing the model.
#'   \item \code{model_getRanges} returns a matrix of size `Nstruct x Ndim` where the rows contain the range values of each basic structure of the model, and the columns to the range values in the principal direction and cross-direction(s) of the basic structure.
#'   \item \code{model_getScales} returns a matrix of size `Nstruct x Ndim` where the rows contain the scale values of each basic structure of the model, and the columns to the scale values in the principal direction and cross-direction(s) of the basic structure.
#'   \item \code{model_getSills} returns an array of size `Nvar x Nvar x Nstruct`: the first two dimensions correspond the covariance matrix between the variables modeled in a given basic structure, and the third dimension corresponds to the different basic structures. For instance, \code{model_getSills(model)[, , 1]} returns the covariance matrix between the `Nvar` variables, as modeled by the first basic structure in the model. If `Nvar=1`, then we retrieve the variance of the only variable modeled by the basic structure.
#'   \item \code{model_getAngles} returns a matrix of size `Nstruct x (Ndim - 1)` whose rows contain the anisotropy angle(s) of each basic structure.
#' }
#' 
#'
#' @return A matrix or vector containing the extracted parameters.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' ## Create model
#' model=createModel(struct=c("EXPONENTIAL", "NUGGET"), range = 0.3, sill = c(1,0.1), ndim=2)
#'
#' ## Extract ranges
#' model_getRanges(model)
#'
#'
model_getStructNames<-function(model){
  nbStruct=model$getNCov()
  nbVar= model$getNVar()
  names=NULL
  for(i in 1:nbStruct){
    names=c(names,model$getCovAniso(i-1)$getCovName())
  }
  return(names)
}


#' @rdname model_getStructNames
#'
#' @export
#'
model_getRanges<-function(model){
  nbStruct=model$getNCov()
  nbVar= model$getNVar()
  ranges=NULL
  for(i in 1:nbStruct){
    ranges=rbind(ranges,model$getCovAniso(i-1)$getRanges())
  }
  rownames(ranges)=model_getStructNames(model)
  
  idNugget=which(rownames(ranges)=="Nugget Effect")
  if(length(idNugget)>0){
    ranges[idNugget,]=NA
  }
  
  return(ranges)
}


#' @rdname model_getStructNames
#'
#' @export
#'
model_getScales<-function(model){
  nbStruct=model$getNCov()
  nbVar= model$getNVar()
  scales=NULL
  for(i in 1:nbStruct){
    scales=rbind(scales,model$getCovAniso(i-1)$getScales())
  }
  rownames(scales)=model_getStructNames(model)
  
  idNugget=which(rownames(scales)=="Nugget Effect")
  if(length(idNugget)>0){
    scales[idNugget,]=NA
  }
  
  return(scales)
}


#' @rdname model_getStructNames
#'
#' @export
#'
model_getSills<-function(model){
  nbStruct=model$getNCov()
  nbVar= model$getNVar()
  sills=array(0,dim=c(nbVar,nbVar,nbStruct))
  for(i in 1:nbStruct){
    sills[,,i]=model$getSills(i-1)$toTL()
  }
  dimnames(sills)[[3]]=model_getStructNames(model)
  
  return(sills)
}


#' @rdname model_getStructNames
#'
#' @export
#'
model_getAnisoAngles<-function(model){
  nbStruct=model$getNCov()
  nbVar= model$getNVar()
  angles=NULL
  for(i in 1:nbStruct){
    angles=rbind(angles,model$getCovAniso(i-1)$getAnisoAngles())
  }
  rownames(angles)=model_getStructNames(model)
  
  idNugget=which(rownames(angles)=="Nugget Effect")
  if(length(idNugget)>0){
    angles[idNugget,]=NA
  }
  
  angles=angles[,1:(ncol(angles)-1)]
  return(angles)
}


.addDriftsToModel<-function(mdl,polDrift=NULL,nExtDrift=0){

  err = mdl$delAllDrifts()
  ## Check if model is coherent with supplied polynomial drift
  if(!is.null(polDrift)){
    ndim=model$getNDim()
  
    if(polDrift>=0){
      err = mdl$addDrift(DriftM())
    }
    if(polDrift>=1){
      indPol=data.frame(matrix(rep(0:polDrift,ndim),ncol=ndim))
      indPol=expand.grid(as.list(indPol))
      indPol=indPol[which(rowSums(indPol)<=polDrift),]
      for(i in 2:nrow(indPol)){
        err = mdl$addDrift(DriftM(indPol[i,]))
      }
    }
  }
  if(nExtDrift>0){
    if(is.null(polDrift)){
      err = mdl$addDrift(DriftM())
    }
    for(i in 0:(nExtDrift-1)){err=mdl$addDrift(DriftF(i))}
  }
}



# struct_eval<-function(h,name="SPHERICAL", range = 0.3, sill = 1, param = 1,mode="VG"){
#   if(mode=="VG"){
#     md=CovCalcMode(); md$setAsVario(TRUE);
#   }else{
#     md=CovCalcMode(); md$setAsVario(FALSE);
#   }
#   cov=CovAniso(ECov_fromKey(name), range, param, sill,CovContext(1,2))
#   res=sapply(h,function(hh){CovAniso_eval(cov,SpacePoint(c(hh,0)),SpacePoint(c(0,0)),mode=md)})
#   return(res)
# }
#



