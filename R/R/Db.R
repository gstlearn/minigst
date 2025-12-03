
#' Function to check if a vector properly defines a set of grid coordinates,
#' i.e. if the values are both sorted in increasing order and regularly spaced
#'
#' @param yseq Vector
#' @param tolRelError Tolerance value to accept that two values are equal
#'
#' @keywords internal
#'
.checkIfRegStep<-function(yseq,tolRelError=1e-6){
  if(length(yseq) >= 3){
    dff=diff(yseq)
    return(((max(diff(dff))/(yseq[2]-yseq[1]) < tolRelError)*(min(dff)>0))==1)
  }else{
    return(TRUE)
  }
}

#' Function to rename all the variables from a Db sharing the same locator
#'
#' @param db Db object
#' @param Loc Locator object
#' @param names Vector of names (must be at least of the same size as the number of variables)
#'
#' @keywords internal
#'
.resetNames<-function(db,Loc,names){
  if(is.null(names)){
    return(invisible(NULL))
  }
  vn=db$getNamesByLocator(Loc)
  if(length(vn)>length(names)){
    stop("The vector of names should be of the same size as (or greater than) the number of variables.")
  }
  for(i in 1:length(vn)){
    db$setName(vn[i],names[i])
  }
  return(invisible(NULL))
}



#' Create an empty DbGrid
#'
#' Function to create an empty \pkg{gstlearn} DbGrid object.
#'
#' @param coords A list containing the coordinates of the grid in each dimension.
#' @param nx A vector containing the size of the grid in each dimension.
#' @param dx A number or a vector specifying the step size of the grid in each dimension. Default = \code{NULL} means that \code{dx=1/(nx-1)}.
#' @param x0 A vector specifying the origin (i.e. lower-left corner) of the grid. Default = \code{NULL} means that \code{x0=rep(0, length(nx))}.
#' @param coordnames A vector specifying names for the grid coordinates. Default = \code{NULL} means that the i-th coordinate will be named xi.
#'
#' @details There are two ways of creating a DbGrid:
#' \itemize{
#'   \item Either by specifying the argument \code{coords} and optionally the argument \code{coordnames}. Then, the dimension of the grid is set by the number of vectors of coordinates given in \code{coords} and the other arguments are ignored.
#'   \item Or by specifying the argument \code{nx} and optionally the arguments \code{dx, x0, coordnames}. Then, the dimension of the grid is set by the length of \code{nx}, and the other arguments are ignored.
#' }
#'
#' @return A DbGrid object.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Define grid points
#' ngrid=100
#' xseq=seq(from=0,to=1,length.out=ngrid)
#' yseq=seq(from=0,to=1,length.out=ngrid)
#'
#' # Create DbGrid
#' dbG=createDbGrid(coords=list(xseq,yseq),coordnames=c("xcoord", "ycoord"))
#' dbG
#'
#' # Alternative way of creating the same DbGrid
#' dbG=createDbGrid(nx=c(ngrid,ngrid),dx=c(1/(ngrid-1),1/(ngrid-1)),x0=c(0,0))
#' dbG
#'
#'
createDbGrid<-function(coords=NULL,nx=NULL,dx=NULL,x0=NULL,coordnames=NULL){
  
  dbG=NULL
  
  if(!is.null(coords)){
    
    if(is.list(coords)){
      
      checkCoords=unlist(lapply(coords, .checkIfRegStep))
      if(prod(checkCoords)!=1){
        stop("Check the coordinates supplied in coords: one or several of them are not sorted in increasing order or not regularly spaced.")
      }
      
      ndim=length(coords)
      if(!is.null(coordnames)){
        if(length(coordnames)!=ndim){
          stop(paste0("The size of the argument coordnames (",length(coordnames),") must be the same size as coords (",ndim,")."))
        }
      }
      
      dbG=DbGrid_create(nx=unlist(lapply(coords, function(v){length(v)})),
                        dx=unlist(lapply(coords, function(v){(v[2]-v[1])})),
                        x0=unlist(lapply(coords, function(v){(v[1])})),
                        flagAddSampleRank=FALSE)
      .resetNames(dbG,ELoc_X(),coordnames)
      
    }else{
      stop("The argument coords must be a list containing the coordinates of the grid points in each dimension.")
    }
  }else{
    
    if(!is.null(nx)){
      
      ndim=length(nx)
      if(!is.null(coordnames)){
        if(length(coordnames)!=ndim){
          stop(paste0("The size of the argument coordnames (",length(coordnames),") must be the same size as nx (",ndim,")."))
        }
      }
      
      if(!is.null(dx)){
        
        if(length(dx)==1){
          dx=rep(dx,ndim)
        }else{
          if(length(dx)!=ndim){
            stop(paste0("The size of the argument dx (",length(dx),") must be 1 or the same size as nx (",ndim,")."))
          }
        }
        
        if(!is.null(x0)){
          
          if(length(x0)!=ndim){
            stop(paste0("The size of the argument x0 (",length(x0),") must be the same size as nx (",ndim,")."))
          }
          dbG=DbGrid_create(nx=nx,dx=dx,x0=x0,flagAddSampleRank=FALSE)
          .resetNames(dbG,ELoc_X(),coordnames)
          
        }else{
          
          dbG=DbGrid_create(nx=nx,dx=dx,flagAddSampleRank=FALSE)
          .resetNames(dbG,ELoc_X(),coordnames)
          
        }
        
      }else{
        
        if(!is.null(x0)){
          
          if(length(x0)!=ndim){
            stop(paste0("The size of the argument x0 (",length(x0),") must be the same size as nx (",ndim,")."))
          }
          dbG=DbGrid_create(nx=nx,x0=x0,flagAddSampleRank=FALSE)
          .resetNames(dbG,ELoc_X(),coordnames)
          
        }else{
          
          dbG=DbGrid_create(nx=nx,flagAddSampleRank=FALSE)
          .resetNames(dbG,ELoc_X(),coordnames)
          
        }
        
      }
    }else{
      stop("You must at least supply the argument coords or the argument nx.")
    }
  }
  
  return(dbG)
}



#' Create a Db or a DbGrid from a dataframe
#'
#' Function to convert a dataframe into a \pkg{gstlearn} Db or DbGrid object.
#'
#' @param df A dataframe containing the data.
#' @param coordnames A vector containing the names of the columns of \code{df} that define the spatial coordinates of the data.
#' @param isGrid Whether the data in \code{df} should be loaded as a set of isolated points (i.e. a Db) or as a grid (i.e. a DbGrid).
#'
#' @details When loading grid data in \code{df}, if the data points are a subset of a grid, a boolean variable \code{Sel} is added to the DbGrid to indicate if each point belong to the initial dataset
#'
#' Calling the function \code{dfToDbGrid} is the same as calling the function \code{dfToDb} with the option \code{isGrid=TRUE}.
#'
#'
#' @return A Db object.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("Scotland")
#' print(head(Scotland))
#'
#' # Create Db of isolated points from dataframe
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db
#'
#' # Load Grid data
#' data("ScotlandGrid")
#' print(head(ScotlandGrid))
#'
#' # Create DbGrid from dataframe
#' dbG=dfToDbGrid(df=ScotlandGrid,coordnames=c("Longitude","Latitude"))
#' dbG
#'
dfToDb<-function(df,coordnames,isGrid=FALSE){
  
  if(class(df)!="data.frame"){
    stop("The argument 'df' must be a dataframe.")
  }
  
  ## Check coordinate names
  var_name=colnames(df)
  if(length(intersect(var_name,coordnames))!=length(coordnames)){
    stop("Check the variable names in coordnames: one or several of the supplied names are absent from the dataframe.")
  }
  
  if(!isGrid){
    
    ## Create Db
    data = Db() # creating the data base
    for (vn in var_name) {data[vn] = suppressWarnings(as.numeric(df[,vn]))}
    ## Set coordinates
    err = data$setLocators(names = coordnames, locatorType = ELoc_X(), cleanSameLocator = TRUE)
    
    return(data)
    
  }else{
    
    tolDiff=1e-6
    
    ndim=length(coordnames)
    coords=list()
    for(i in 1:ndim){
      coords[[i]]=sort(unique(as.vector(df[,coordnames[i]])))
    }
    checkCoords=unlist(lapply(coords, .checkIfRegStep))
    if(prod(checkCoords)!=1){
      stop("The grid coordinates supplied in df are not regularly spaced.")
    }
    
    # Create DbGrid
    dbG=createDbGrid(coords=coords,coordnames = coordnames)
    if (dbG$getNSample() == nrow(df)) {
      if(max(abs(dbG[,coordnames]-df[,coordnames]))<tolDiff){
        
        ## The coordinates coincide, we can simply copy the variables
        var_name=setdiff(colnames(df),coordnames)
        for (vn in var_name) {dbG[vn] = suppressWarnings(as.numeric(unlist(df[,vn])))}
        
      }else{
        warnings("The coordinates in df and in the Db are not arranged in the same order. Consequently, the variables from df are migrated to the closest grid point in the Db.")
        
        ## Create Db
        data = Db() # creating the data base
        for (vn in var_name) {data[vn] = suppressWarnings(as.numeric(df[,vn]))}
        ## Set coordinates
        err = data$setLocators(names = coordnames, locatorType = ELoc_X(), cleanSameLocator = TRUE)
        
        DbGrid_migrateAllVariables(data,dbG,flagAddSampleRank = FALSE)
      }
    }else{ #if the gridded points are not a discretization of a rectangular parallelepiped
      data = Db() # creating the data base
      for (vn in var_name) {data[vn] = suppressWarnings(as.numeric(df[,vn]))}
      ## Set coordinates
      err = data$setLocators(names = coordnames, locatorType = ELoc_X(), cleanSameLocator = TRUE)
      ddx = NULL
      for(i in 1:ndim){
        ddx=c(ddx,unique(diff(coords[[i]])))
      }
      dbG = DbGrid_createCoveringDb(data,dx = ddx)
      for(i in 1:ndim){
        err=dbG$setName(paste0("x",i),coordnames[i])
      }
      err = migrateMulti(dbin=data, dbout = dbG,  names = names(df)[(ndim+1):length(var_name)], namconv=NamingConvention())
      err=Db_clearLocators(dbG,ELoc_Z())
      # gérer la sélection
      res = dbG$addSelection(tab = !is.na(dbG[var_name[ndim+1]]), name = "Sel_Not_NA")
    }
    return(dbG)
    
  }
  
}


#' @rdname dfToDb
#'
#' @export
#'
dfToDbGrid<-function(df,coordnames){
  
  return(dfToDb(df,coordnames,isGrid=TRUE))
  
}




#----------------------------------------------


#' Manage variables in a Db
#'
#' Functions to add  new variables/columns to an existing `gstlearn` Db object, or delete existing variables/columns from a Db.
#'
#' @param db Db object.
#' @param var A vector (or a matrix) containing the new variable(s) to be added to \code{db}. The length (or number of rows) of `var` must be equal to the number of points/rows in \code{db} (given by \code{nrow(db[])}).
#' @param vname Name(s) of the new variables to add or of the existing variables to delete, stored as a (vector of) string(s).
#'
#' @return The function updates \code{db} and returns nothing.
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
#' # Create 2 new variables in a matrix
#' newVars=matrix(rnorm(2*nrow(db[])),ncol=2)  # Matrix of normal values
#' head(newVars)
#'
#' # Add variables to db
#' addVarToDb(db=db, var = newVars, vname = c("Newvar1", "Newvar2"))
#' db$display()
#' 
#' # Delete variables from db
#' delVarFromDb(db=db, vname = c("Newvar1", "Newvar2"))
#' db$display()
#' 
#'
addVarToDb<-function(db,var,vname){
  
  if(is.null(dim(var))){
    N=length(var)
    var=as.matrix(var,ncol=1)
  }else{
    N=nrow(var)
  }
  
  if(N!=nrow(db[])){
    stop("The variables should have the same length (",N,") as the number of points in the Db (",nrow(db[]),").")
  }
  if(length(vname)!=ncol(var)){
    stop("The number of variable names (",length(vname),") should be the same as the number of variables (",ncol(var),").")
  }
  
  for(i in 1:length(vname)){
    db[vname[i]]=var[,i]
  }
  
  return(invisible(NULL))
  
}


#' @rdname addVarToDb
#'
#' @export
#'
delVarFromDb<-function(db,vname){
  for(v in vname){
    if(v %in% colnames(db[])){
      db$deleteColumns(names = v)
    }
  }
  return(invisible(NULL))
}



#----------------------------------------------


#' Add a selection to a Db
#'
#' Function to add a new selection/mask to an existing \pkg{gstlearn} Db object.
#'
#' @param db Db object.
#' @param sel A vector of boolean values indicating which points of \code{db} are selected. The length of \code{sel} must be equal to the number of points/rows in \code{db} (given by \code{nrow(db[])}).
#'
#' @return The function updates \code{db} and returns nothing.
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
#' # Select points with an elevation greater than 100
#' selVar=(db["Elevation"]>100)
#'
#' # Add selection to Db
#' addSel(db=db, sel = selVar)
#' db$display()
#'
#' # Clear the selection from the Db
#' clearSel(db=db)
#' db$display()
#'
addSel<-function(db,sel){
  if(length(sel)!=nrow(db[])){
    stop(paste0("The size of the selection vector (",length(sel), ") should be the same as the number of points in the Db (",nrow(db[]),")"))
  }
  db$deleteColumns(names = "Selection")
  err=db$addSelection(tab = sel, name = "Selection")
  return(invisible(NULL))
}

#' @rdname addSel
#'
#' @export
#'
clearSel<-function(db){
  db$deleteColumnsByLocator(ELoc_SEL())
  return(invisible(NULL))
}



#----------------------------------------------


#' Set variables of interest in a Db
#'
#' Function to select which variables in the Db are the variables of interest or drifts in the study.
#'
#' @param db Db object.
#' @param vname Name(s) of the selected variable(s), stored as a (vector of) string(s).
#' @param mode Role of the variable: either "Var" for variable of interest, or "Drift" for drift specification.
#'
#' @return The function updates \code{db} and returns nothing.
#'
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' 
#'
#' # Create Db
#' data("Scotland")
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#' db$display()
#'
#' # Select the elevation as the variable of interest
#' setVar(db=db, vname="Elevation")
#' db$display()
#'
setVar<-function(db,vname,mode="Var"){
  if(mode=="Var"){
    db$clearLocators(ELoc_Z())
    if(length(intersect(colnames(db[]),vname))==0){
      stop("Check the variable names: one or several of the supplied names are absent from the Db.")
    }
    err = db$setLocators(name = vname, locatorType = ELoc_Z(), cleanSameLocator = TRUE)
  }
  if(mode=="Drift"){
    db$clearLocators(ELoc_F())
    if(length(intersect(colnames(db[]),vname))==0){
      stop("Check the variable names: one or several of the supplied names are absent from the Db.")
    }
    err = db$setLocators(name = vname, locatorType = ELoc_F(), cleanSameLocator = TRUE)
  }

  return(invisible(NULL))
}


#----------------------------------------------



#' Compute summary statistics
#'
#' Function to compute summary statistics on the variables of a Db.
#'
#' @param db Db object.
#' @param vname Name(s) of the variable(s) for which statistics are desired, stored as a (vector of) string(s).
#' @param stat Vector containing the names of the statistics to compute. Available choices are listed the section \emph{Details} below. The default value (\code{NULL}) computes all the available statistics.
#' @param onlyCommon Compute the statistics based only on the points for which all variables are defined (i.e. are not \code{NA}).
#'
#' @return A dataframe containing the selected statistics computed for each specified variable.
#'
#' @details The available choices for the statistics are as follows:
#' \describe{
#'    \item{\code{"NUM"}}{Number of defined values}
#'    \item{\code{"MIN"}}{Minimum}
#'    \item{\code{"MAX"}}{Maximum}
#'    \item{\code{"MEAN"}}{Mean}
#'    \item{\code{"STDV"}}{Standard deviation}
#'    \item{\code{"MED"}}{Median}
#'    }
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
#'
#' # Compute summary statistics
#' sumStats=summaryStats(db = db, vname = c("Elevation","January_temp"))
#' print(sumStats)
#'
summaryStats<-function(db,vname,stat=NULL,onlyCommon=FALSE){
  
  if(length(intersect(colnames(db[]),vname))<length(vname)){
    stop("Check the variable names: one or several of the supplied names are absent from the Db.")
  }
  if(length(stat)<=0){
    stat=c("NUM","MIN","MAX","MEAN","STDV","MED")
  }
  
  if(onlyCommon){
    id_dat=which(!is.na(rowSums(db[,vname])))
  }else{
    id_dat=1:nrow(db[])
  }
  
  if(length(db$getNamesByLocator(ELoc_SEL())>0)){
    id_dat=intersect(id_dat,which(apply(as.matrix(db[db$getNamesByLocator(ELoc_SEL())]),1,prod)==1))
  }
  
  
  res=NULL
  nres=NULL
  for(sname in stat){
    st=switch(sname,
              NUM = {apply(db[id_dat,vname],2,function(v){sum(!is.na(v))})},
              MIN = {apply(db[id_dat,vname],2,function(v){min(v,na.rm=T)})},
              MAX = {apply(db[id_dat,vname],2,function(v){max(v,na.rm=T)})},
              MEAN = {apply(db[id_dat,vname],2,function(v){mean(v,na.rm=T)})},
              STDV = {apply(db[id_dat,vname],2,function(v){sd(v,na.rm=T)})},
              MED = {apply(db[id_dat,vname],2,function(v){median(v,na.rm=T)})})
    
    if(!is.null(st)){
      nres=c(nres,sname)
      res=cbind(res,st)
    }
  }
  colnames(res)=nres
  return(as.data.frame(res))
  
}



#'
#' Function to delete variable which already exists in a Db
#'
#' @keywords internal
#'
.checkVariableNames<-function(db,vname){
  for(v in vname){
    if(!(v %in% colnames(db[]))){
      stop("The variable '",v,"' is absent from the Db. Please check the name.")
    }
  }
  return(invisible(NULL))
}



#'
#' Migrate variables from one Db to another one
#'
#' Function to migrate variables from one Db to another one (by assigning to each location of the target Db the values of the variables at the closest location in the input Db).
#' 
#' @param dbin Db object containing the data from which the migration takes place.
#' @param dbout Db object containing the data to which the migration takes place.
#' @param vname Names of the variables (in `dbin`) to migrate: if `NULL`, all the variables are migrated.
#' @param prefix Prefix added to the migrated variables (in `dbout`)
#' @param addIdOrigin Add a column in `dbout` to indicate the id (row number in `dbin`) of the migrated variable.
#' @param addSelNotNA Add a selection in `dbout` to select only the points where a variable was migrated.
#'
#' @return Updates `dbout` by adding the migrated variables and returns nothing.
#'
#' @export
#'
#' @examples
#' library(minigst)
#' 
#' ## Load Data
#' data("Scotland")
#' data("ScotlandGrid")
#' 
#' ## Origin and destination Db
#' dbin=dfToDb(Scotland,c("Longitude","Latitude"),isGrid = F)
#' dbout=dfToDbGrid(ScotlandGrid,c("Longitude","Latitude"))
#' 
#' ## Migration
#' migrateVar(dbin,dbout,prefix="Migr")
#' dbout
#'
migrateVar<-function(dbin,dbout,vname=NULL,prefix=NULL,
                     addIdOrigin=TRUE,addSelNotNA=FALSE){
  
  ## While waiting for issues with migrate function to be solved
  dbin_t=Db(dbin)
  dbout_t=Db(dbout)
    
  dbin_t["idOrigin"]=1:nrow(dbin_t[])
  
  if(is.null(vname)){
    vnames=setdiff(colnames(dbin_t[]),dbin_t$getNamesByLocator(ELoc_X()))
  }else{
    .checkVariableNames(dbin_t,vname)
    vnames=vname
  }
  
  if(is.null(prefix)|| !is.character(prefix)){
    delVarFromDb(dbout_t,c(vnames,"Sel_Not_NA"))
    err = migrateMulti(dbin = dbin_t, dbout = dbout_t,  names = vnames,
                       namconv = NamingConvention())
    if(addSelNotNA){
      res = dbout_t$addSelection(tab = !is.na(dbout_t[]["idOrigin"]), name = "Sel_Not_NA")
    }
    if(!addIdOrigin){
      delVarFromDb(dbout_t,"idOrigin")
    }
  }else{
    delVarFromDb(dbout_t,c(paste0(prefix,".",vnames),"Sel_Not_NA"))
    err = migrateMulti(dbin =dbin_t, dbout = dbout_t,  names = vnames,
                       namconv=NamingConvention(prefix))
    if(addSelNotNA){
      res = dbout_t$addSelection(tab = !is.na(dbout_t[][paste0(prefix,".","idOrigin")]), name = "Sel_Not_NA")
    }
    if(!addIdOrigin){
      delVarFromDb(dbout_t,paste0(prefix,".","idOrigin"))
    }
  }
  
  ## Clean
  err = Db_clearLocators(dbout_t,ELoc_Z())
  delVarFromDb(dbin_t,"idOrigin")
  
  ## Add migrated variables to dbout
  new_vars=setdiff(names(dbout_t[]),names(dbout[]))
  if(length(new_vars)>0){
    for (new in new_vars) {
      addVarToDb(dbout,dbout_t[new],new)
    }
  }
  
}

