#' Function to create breaks (used to plot size legend in `dbplot_point`)
#' @keywords internal
#' 
getSeq<-function(v,n=4,nd=3){
  u=floor(seq(from=floor(v[1]*10**nd),to=ceiling(v[2]*10**nd),length.out=n))/10**nd
  u=pretty(v,n)
  return(u)
}

#' Plot variables in a Db
#'
#' Function to plot variables in a Db object as isolated points. The value of the variables are used to define
#' the color, size and/or labels of the points.
#'
#' @param db Db object.
#' @param color,cat_color Name of the variable defining the color of the points: use `color` for continuous variables and `cat_color` for categorical variables.
#' @param size Name of the variable defining the size of the points.
#' @param cmap Color map used to plot the points by color. Can be specified by a name (cf. \code{\link{printAllPalettes}} to display available choices), or by a vector of color names which will be interpolated to create a palette.
#' @param colValLimits Range of values for the color mapping.
#' @param naColor Color used for NA values.
#' @param pointColor Default color of points when \code{color} is not specified.
#' @param pointSize Default size of points when \code{size} is not specified.
#' @param sizeRange Vector specifying the range of sizes when plotting the points by size.
#' @param absSize Whether the absolute value of the variable should be used to define the size of the points.
#' @param hideLegend Whether to hide the legend.
#' @param legendTitle Title(s) of the legend: either a single string if just the color or the size if plotted, or a vector containing respectively the title of the color legend and the title of the size legend. Setting the legend title to `NA` means that the name of the variable is used as a title.
#' @param asp Aspect ratio of the plot: 0 = No scaling of the axes, 1 = Axes have the same scale.
#' @param xlab Title of the x-axis.
#' @param ylab Title of the y-axis.
#' @param title Title of plot.
#' @param add Either a boolean or \pkg{ggplot2} object. If \code{FALSE} (Default), then a new plot is created. If \code{TRUE}, then the points are added to the last plot (see \pkg{ggplot2::last_plot()}) specifying if the points are to be added to the last plot or not
#'
#' @details Note that you can specify only one of the arguments `color` and `cat_color`,but you can specify it jointly with the `size` argument.
#' If none of these variables is specified, the points are all plotted with a common size `pointSize` and color `pointColor`.
#'
#' @return A \pkg{ggplot2} object containing the plot. If stored in a variable, use the function \code{print} to display the plot.
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
#' # Plot the variables Elevation (for the size of the points) and Longitude (for the color of the points)
#' dbplot_point(db=db,size="Elevation",color="Longitude")
#'
#' # Store the plot, and then display it
#' plt=dbplot_point(db=db,size="Elevation",title="Printed plot")
#' print(plt)
#'
dbplot_point<-function(db,color=NULL,cat_color=NULL,size=NULL,
                       cmap=NULL,colValLimits=NULL,naColor="transparent",
                       pointColor="black",pointSize=1,
                       sizeRange=c(0.5,3),absSize=FALSE,
                       hideLegend=FALSE, legendTitle=NA,
                       asp = 1, xlab = NULL, ylab = NULL, title = NULL, add = FALSE){
  
  .checkVariableNames(db,c(color,cat_color,size))
  
  if(is.logical(add)){
    if(add){ 
      p = last_plot()
    }else{
      p = plot.init()
    }
  }else{
    p = add
  }
  
  df=db[]
  xname=db$getNamesByLocator(ELoc_X())
  selnames=db$getNamesByLocator(ELoc_SEL())
  if(length(selnames)>0){
    selvec=(apply(as.matrix(df[selnames]),1,prod)==1)
    df=df[selvec,]
  }
  
  aes_plt=aes(x=.data[[xname[1]]],y=.data[[xname[2]]])
    
  if(!is.null(size)){
    if(absSize){
      aes_plt$size = substitute(abs(.data[[size]]))
    }else{
      aes_plt$size = substitute(.data[[size]])
    }
  }

  discreteVal=FALSE
  if(!is.null(color)){
    aes_plt$colour = substitute(.data[[color]])
  }
  
  if(!is.null(cat_color)){
    if(!is.null(color)){
      stop("You can either specify the `color` argument or the `cat_color` argument, but not both.")
    }else{
      discreteVal=TRUE
      aes_plt$colour = substitute(factor(.data[[cat_color]]))
      ncol=length(levels(factor(df[[cat_color]])))
      if(is.null(cmap)){
        cmap=1:ncol
      }else if(length(cmap)<ncol){
        stop(paste0("When using discrete color maps (`discreteVal=TRUE`), the argument `cmap` must be either `NULL` or a vector of ",ncol," colors. However, only ",length(cmap)," colors were specified."))
      }
    }
  }

  if(is.null(color)&&is.null(cat_color)){
    if(is.null(size)){
      p = p + geom_point(data=df,aes_plt,color=pointColor,
                         size=pointSize,
                         show.legend=ifelse(hideLegend,FALSE,NA))
    }else{
      p = p + geom_point(data=df,aes_plt,color=pointColor,
                         show.legend=ifelse(hideLegend,FALSE,NA))
    }
  }else{
    if(is.null(size)){
      p = p + geom_point(data=df,aes_plt,size=pointSize,show.legend=ifelse(hideLegend,FALSE,NA))
    }else{
      p = p + geom_point(data=df,aes_plt,show.legend=ifelse(hideLegend,FALSE,NA))
    }
  }
  
  if(!is.null(size)){
    
    if(!is.null(color)||!is.null(cat_color)){
      sizeLegendTitle=legendTitle[2]
    }else{
      sizeLegendTitle=legendTitle[1]
    }
    name = .defineName(sizeLegendTitle)
    
    id_not_NA=which(!is.na(df[,size]))
    df=df[id_not_NA,]
    if(absSize){
      valSize=abs(df[,size])
    }else{
      valSize=df[,size]
    }
    breaks = getSeq(range(valSize,na.rm=TRUE))
    p = p + scale_size(breaks = breaks,limits=range(breaks),range = sizeRange,name=name)
    p = p + new_scale("size")
  }

  if(!is.null(color)||!is.null(cat_color)){
    colorLegendTitle=legendTitle[1]
    p = p + .defineColour(cmap,flagDiscrete=discreteVal,naColor = naColor,limits=colValLimits,title = colorLegendTitle)
    p = p + new_scale("colour")
  }

  p = p + plot.geometry(asp=asp)
  if(is.null(xlab)){
    xlab=xname[1]
  }
  if(is.null(ylab)){
    ylab=xname[2]
  }
  if(is.null(title)){
    p = p + plot.decoration(xlab = xlab, ylab = ylab, title = "")
  }else{
    p = p + plot.decoration(xlab = xlab, ylab = ylab, title = title)
  }
  
  return(p)
}

#' Add lines to an existing plot
#'
#' Function to add lines to an existing plot.
#'
#' @param plt A \pkg{ggplot2} plot object. 
#' @param a,b Slope and intercept of the line (single values).
#' @param h Value defining an horizontal line.
#' @param v Value defining a vertical line.
#' @param color Name of the color of the line.
#' @param linetype Type of line, specified by either an integer (0-6) or a name (0 = "blank", 1 = "solid", 2 = "dashed", 3 = "dotted", 4 = "dotdash", 5 = "longdash", 6 = "twodash").
#'
#' @return A \pkg{ggplot2} object containing the updated plot.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data and create Db
#' data("Scotland")
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#'
#' # Create a plot of  the variable Elevation
#' plt=dbplot_point(db=db,size="Elevation")
#'
#' # Add  a vertical line to the plot
#' plt=addLines(plt=plt,v=300,color="orange")
#'
#' # Add a  line y = 5 * x +100
#' plt=addLines(plt=plt,a=5,b=100,color="blue",linetype=2)
#'
#' # Display the plot
#' print(plt)
#'
#' ## Alternatively, without using print (creates a new, updated plot each time)
#' dbplot_point(db=db,size="Elevation")
#' addLines(v=300, color = "orange")
#' addLines(a=5,b=100,color="blue",linetype=2)
#'

addLines<-function(plt=last_plot(),a=NULL,b=NULL,h=NULL,v=NULL,color="black",linetype=1){

  if((!is.null(a))&&(!is.null(b))){
    plt=plt+geom_abline(slope = a,intercept = b,color=color,linetype=linetype)
  }
  if(!is.null(h)){
    plt=plt+geom_hline(yintercept = h,color=color,linetype=linetype)
  }
  if(!is.null(v)){
    plt=plt+geom_vline(xintercept = v,color=color,linetype=linetype)
  }
  return(plt)

}

#' Add points to an existing plot
#'
#' Function to add points to an existing plot.
#'
#' @param plt A \pkg{ggplot2} plot object.
#' @param x,y Coordinates of the points.
#' @param color Name of the color of the points.
#' @param shape Type of point, specified by either an integer (0-25) or a name, see \code{\link{vignette("ggplot2-specs")}}
#' @param size Size of the point in mm.
#'
#' @return A \pkg{ggplot2} object containing the updated plot.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data and create Db
#' data("Scotland")
#' db=dfToDb(df=Scotland, coordnames=c("Longitude", "Latitude"))
#'
#' # Create a plot of  the variable Elevation
#' plt=dbplot_point(db=db,size="Elevation")
#'
#' # Add two triangular-shaped, orange points the plot
#' plt=addPoints(plt=plt,x=c(100,400),y=c(600,1100),color="orange", shape = 17, size = 3)
#'
#' # Display the plot
#' print(plt)
#'
#'
addPoints<-function(plt=last.plot(),x=NULL,y=NULL,color="black", shape = 16, size = 1){
  plt = plt + geom_point(aes(x=x,y=y), colour = color, shape = shape, size = size)
  return(plt)
}


#' Plot variables in a Db
#'
#' Function to plot variables in a Db object as isolated points. The value of the variables are used to define
#' the color, size and/or labels of the points.
#'
#' @param dbGrid DbGrid object.
#' @param color,cat_color Name of the variable defining the color of the points: use `color` for continuous variables and `cat_color` for categorical variables.
#' @param contour Name of the variable defining the contours of the plot.
#' @param cmap Color map used to plot the points by color. Can be specified by a name (cf. \code{\link{printAllPalettes}} to display available choices), or by a vector of color names which will be interpolated to create a palette.
#' @param colValLimits Range of values for the color mapping.
#' @param naColor Color used for NA values.
#' @param nLevels Number of levels for the contour plot.
#' @param hideLegend Whether to hide the legend.
#' @param legendTitle Title of the legend.
#' @param asp Aspect ratio of the plot: 0 = No scaling of the axes, 1 = Axes have the same scale.
#' @param xlab Title of the x-axis.
#' @param ylab Title of the y-axis.
#' @param title Title of plot.
#' @param add Either a boolean or \pkg{ggplot2} object. If \code{FALSE} (Default), then a new plot is created. If \code{TRUE}, then the points are added to the last plot (see \pkg{ggplot2::last_plot()}) specifying if the points are to be added to the last plot or not
#'
#' @details Note that you can specify only one of the arguments `color`, `cat_color` and `contour`.
#'
#' @return A \pkg{ggplot2} object containing the plot. If stored in a variable, use the function \code{print} to display the plot.
#'
#' @export
#'
#' @examples
#' library(minigst)
#'
#' # Load Data
#' data("ScotlandGrid")
#'
#' # Create DbGrid
#' dbG=dfToDbGrid(df=ScotlandGrid,coordnames = c("Longitude","Latitude"))
#'
#' # Plot the variables Elevation by color using the "viridis" palette
#' dbplot_grid(dbG,color="Elevation",cmap = "viridis")
#'
#' # Store the plot, and then display it
#' plt= dbplot_grid(dbG,color="Elevation",cmap = "viridis",title="Printed plot")
#' print(plt)
#'
dbplot_grid<-function(db,color=NULL,cat_color=NULL,contour=NULL,
                       cmap=NULL,colValLimits=NULL,naColor="transparent",
                       nLevels=NULL,
                       hideLegend=FALSE, legendTitle=NA,
                       asp = 1, xlab = NULL, ylab = NULL, title = NULL, add = FALSE)
{
  if((!is.null(color))+(!is.null(cat_color))+(!is.null(contour))>1){
    stop("You must specify only one of the following arguments: color, cat_color, contour")
  }
  .checkVariableNames(db,c(color,cat_color,contour))
  
  if(is.logical(add)){
    if(add){
      p = last_plot()
    }else{
      p = plot.init()
    }
  }else{
    p = add
  }
  
  df=db[]
  ## Extract name of coordinates
  xname0=db$getNamesByLocator(ELoc_X())
  ## Copy coordinates 
  xname=paste0(xname0,"_copy")
  df[,xname]=df[,xname0]
  ## Extarct and apply selection
  selnames=db$getNamesByLocator(ELoc_SEL())
  if(length(selnames)>0){
    selvec=apply(as.matrix(df[selnames]),1,prod)
    selvec[selvec==0]=NA
    df[,setdiff(colnames(df),xname)]=df[,setdiff(colnames(df),xname)]*selvec
  }
  
  aes_plt=aes(x=.data[[xname[1]]],y=.data[[xname[2]]])
  
  discreteVal=FALSE
  if(!is.null(color)){
    aes_plt$fill = substitute(.data[[color]])
  }
  if(!is.null(cat_color)){
    if(!is.null(color)){
      stop("You can either specify the `color` argument or the `cat_color` argument, but not both.")
    }else{
      discreteVal=TRUE
      aes_plt$fill = substitute(factor(.data[[cat_color]]))
      ncol=length(levels(factor(df[[cat_color]])))
      if(is.null(cmap)){
        cmap=1:ncol
      }else if(length(cmap)<ncol){
        stop(paste0("When using discrete color maps (`discreteVal=TRUE`), the argument `cmap` must be either `NULL` or a vector of ",ncol," colors. However, only ",length(cmap)," colors were specified."))
      }
    }
  }
  
  if(!is.null(color)||!is.null(cat_color)){
    p = p + geom_raster(data=df,aes_plt,show.legend=ifelse(hideLegend,FALSE,NA)) 
    p = p + .defineFill(cmap,flagDiscrete=discreteVal,naColor = naColor,limits=colValLimits,title = legendTitle)
    p = p + new_scale("fill")
  }else if(!is.null(contour)){
    p = p + geom_contour(data=df,
                     aes(x=.data[[xname[1]]],y=.data[[xname[2]]],
                         z=.data[[contour]],colour=after_stat(level)
                     ),
                     bins = nLevels,
                     na.rm=TRUE,show.legend=ifelse(hideLegend,FALSE,NA))
    p = p + .defineColour(cmap,title = legendTitle)
    p = p + new_scale("colour")
  }else{
    stop("You must specify one of the following arguments: color, cat_color, contour")
  }
  
  p = p + plot.geometry(asp=asp)
  if(is.null(xlab)){
    xlab=xname0[1]
  }
  if(is.null(ylab)){
    ylab=xname0[2]
  }
  if(is.null(title)){
    p=p+plot.decoration(xlab = xlab, ylab = ylab,title = "")
  }else{
    p=p+plot.decoration(xlab = xlab, ylab = ylab, title = title)
  }
  
  return(p)
}

#------------------------------------------------------------

#' Function to plot a single experimental variogram or cross-variogram and overlay a model on top.
#'
#' See \code{plot_vario} for more info on the parameters.
#'
#' @keywords internal
#'
.plot_vario_base<-function(expvario=NA,
                     model=NA,
                     ivar=0, jvar=0, idir=-1,
                     pairDisplay=NA,
                     nlag = 100, hmax = NA,
                     color=NULL, title="", legend=TRUE, ...){

  nvar=expvario$getNVar()
  ndir=expvario$getNDir()
  if(ivar<(-1) || ivar >= nvar){
    stop(
      paste0("The index 'ivar' should be between -1 and ",nvar-1,".")
    )
  }
  if(jvar<(-1) || jvar >= nvar){
    stop(
      paste0("The index 'jvar' should be between -1 and ",nvar-1,".")
    )
  }
  if(idir<(-1) || idir >= ndir){
    stop(
      paste0("The index 'idir' should be between -1 and ",ndir-1,".")
    )
  }

  if((!is.na(pairDisplay))&&(pairDisplay=="size")){
    drawPsize=T
    drawPlabel=F
  }else if((!is.na(pairDisplay))&&(pairDisplay=="label")){
    drawPsize=F
    drawPlabel=T
  }else{
    drawPsize=-1
    drawPlabel=F
  }

  p = plot.init()
  if(is.null(color)){
    p=p+suppressWarnings(plot.varmod(vario=expvario,model=model,nh=nlag,hmax=hmax,drawPsize = drawPsize,
                    ivar=ivar, jvar=jvar, idir=idir,
                    drawPlabel = drawPlabel,flagLegend = legend,flagEnvelop = F,...))
  }else{
    p=p+suppressWarnings(plot.varmod(vario=expvario,model=model,nh=nlag,hmax=hmax,drawPsize = drawPsize,
                    ivar=ivar, jvar=jvar, idir=idir,
                    drawPlabel = drawPlabel,flagLegend = legend,flagEnvelop = F,color=color,...))
  }

  p=p+plot.decoration(xlab = "Distance", ylab="Variogram", title=title)
  return(p)
}

.short_str <- function(str_vec,n=12) {
  res=NULL
  for (i in 1:length(str_vec)) {
    str=str_vec[i]
    nstr=nchar(str)
    if(nstr>n){
      ni=(n %/% 2) -1
      res=c(res,paste0(substr(str,1,ni),"..",substr(str,nstr-ni,nstr)))
    }else{
      res=c(res,str)
    }
  }
  return(res)
}

#' Plot experimental variograms and cross-variograms.
#'
#' Function to plot experimental variograms and cross-variograms and overlay a model on top.
#'
#'
#' @param expvario Experimental variogram(s) (as a \pkg{gstlearn} object).
#' @param model Model (as a \pkg{gstlearn} object).
#' @param ivar Rank of the variable to be represented: single value between 0 and (Number of variables -1), or -1 for all variables
#' @param jvar Rank of the second variable to be represented in multivariate case: single value between 0 and (Number of variables -1), or -1 for all variables
#' @param idir Rank of the direction to be represented: single value between 0 and (Number of directions -1), or -1 for all directions
#' @param pairDisplay Display information about the number of pairs used to compute the variogram values: "size" = Size of points, "label" = As a label, NA = No information displayed.
#' @param nlag Number of distance lags (used if 'expvario' is not defined)
#' @param hmax Maximum distance (used if 'expvario' is not defined)
#' @param color Single color to use for all curves.
#' @param title Title of plot.
#' @param legend Whether to plot the legend.
#'
#'
#' @return A \pkg{ggplot2} object containing the plot. If stored in a variable, use the function \code{print} to display the plot.
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
plot_vario<-function(expvario=NA,
                     model=NA,
                     ivar=-1, jvar=-1, idir=-1,
                     pairDisplay=NA,
                     nlag = 100, hmax = NA,
                     color=NULL, title="", legend=TRUE){

  nvar=expvario$getNVar()
  ndir=expvario$getNDir()
  if(ivar<(-1) || ivar >= nvar){
    stop(
      paste0("The index 'ivar' should be between -1 and ",nvar-1,".")
    )
  }
  if(jvar<(-1) || jvar >= nvar){
    stop(
      paste0("The index 'jvar' should be between -1 and ",nvar-1,".")
    )
  }
  if(idir<(-1) || idir >= ndir){
    stop(
      paste0("The index 'idir' should be between -1 and ",ndir-1,".")
    )
  }

  if((ivar==-1)&&(jvar==-1)&&(nvar>1)){

    ## Case i=-1, j=-1, nvar>1
    pl=list()
    k=1
    for(i in 1:nvar -1){
      for(j in 1:nvar -1){
        if(j<=i){
          vnames=.short_str(expvario$getVariableNames())
          pl[[k]]=.plot_vario_base(expvario=expvario,model=model,ivar = i,jvar=j,idir=idir,
                             pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                             # title =paste0("(",i,",",j,")"),
                             title =paste0(vnames[i+1],"/",vnames[j+1]),
                             legend = F)
        }else{
          if((j==nvar-1)&&(i==0)&&legend){
            leg=get_legend(.plot_vario_base(expvario=expvario,model=model,ivar = i,jvar=j,idir=idir,
                                      pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                                      #title =paste0("(",i,",",j,")"),
                                      title =paste0(vnames[i+1],"/",vnames[j+1]),
                                      legend = T))
            pl[[k]]=as_ggplot(leg)
          }else{
            pl[[k]]=plot.init()+theme(
              panel.background = element_rect(fill='transparent'), #transparent panel bg
              plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
              panel.grid.major = element_blank(), #remove major gridlines
              panel.grid.minor = element_blank(), #remove minor gridlines
              legend.background = element_rect(fill='transparent'), #transparent legend bg
              legend.box.background = element_rect(fill='transparent')) #transparent legend panel
          }
        }
        k=k+1
      }
    }
    p=ggarrange(plotlist = pl,ncol=nvar,nrow=nvar)
    if(!is.null(title)){
      p=annotate_figure(p, top = text_grob(title, face = "bold", size = 14))
    }

  }else if((jvar==-1)&&(nvar>1)){

    ## Case i=ivar, j=-1, nvar>1
    pl=list()
    k=1
    for(j in 1:nvar-1){
      pl[[k]]=.plot_vario_base(expvario=expvario,model=model,ivar = ivar,jvar=j,idir=idir,
                         pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                         title =paste0("(",ivar,",",j,")"),
                         legend = F)
      k=k+1
    }
    if(legend){
      leg=get_legend(.plot_vario_base(expvario=expvario,model=model,ivar = ivar,jvar=j,idir=idir,
                                pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                                title =paste0("(",ivar,",",j,")"),
                                legend = T))
      pl[[k]]=as_ggplot(leg)
    }
    p=ggarrange(plotlist = pl,ncol=length(pl),nrow=1)
    if(!is.null(title)){
      p=annotate_figure(p, top = text_grob(title, face = "bold", size = 14))
    }

  }else if((ivar==-1)&&(nvar>1)){

    pl=list()
    k=1
    for(i in 1:nvar-1){
      pl[[k]]=.plot_vario_base(expvario=expvario,model=model,ivar = i,jvar=jvar,idir=idir,
                         pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                         title =paste0("(",i,",",jvar,")"),
                         legend = F)
      k=k+1
    }
    if(legend){
      leg=get_legend(.plot_vario_base(expvario=expvario,model=model,ivar = i,jvar=jvar,idir=idir,
                                pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                                title =paste0("(",i,",",jvar,")"),
                                legend = F)+theme(legend.position="bottom"))
      pl[[k]]=as_ggplot(leg)
    }
    p=ggarrange(plotlist = pl,nrow=length(pl),ncol=1)
    if(!is.null(title)){
      p=annotate_figure(p, top = text_grob(title, face = "bold", size = 14))
    }

  }else if(nvar ==1){
    p=.plot_vario_base(expvario=expvario,model=model,ivar = 0,jvar=0,idir=idir,
               pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
               title = title,
               legend = legend)
  }else{
    p=.plot_vario_base(expvario=expvario,model=model,ivar = ivar,jvar=jvar,idir=idir,
                 pairDisplay=pairDisplay, nlag = nlag, hmax = hmax, color=color,
                 title =title,
                 legend = legend)
  }

  return(p)
}


#' Plot histogram of a variable in a gstlearn db
#'
#' Function to plot the histogram of a variable in a gstlearn db
#'
#'
#' @param db Db object.
#' @param vname name of the variable to be represented
#' @param color Single color to use for all curves (default to blue)
#' @param nbin a single number giving the number of cells for the histogram
#' @param title Title of plot
#' @param xlab label of the abscissa
#' 
#' @return A \pkg{ggplot2} object containing the plot. If stored in a variable, use the function \code{print} to display the plot.
#' 
#' @export
#'
#' 
plot_hist<-function (db,vname,color = "blue",nbin=30, title=NULL,xlab=NULL){
  p = plot.init()
  p = p + plot.hist(db,name=vname,bins = nbin,fill=color)
  p = p + plot.decoration(xlab=xlab, title=title)
  return(p)
}

#' Plot correlation plot of a couple of variables in a gstlearn db
#'
#' Function to plot the histogram of a variable in a gstlearn db
#'
#'
#' @param db Db object.
#' @param vnamex Name of the variable to be represented in abscissa
#' @param vnamey Name of the variable to be represented in ordinate
#' @param xlab Label of x-axis
#' @param ylab Label of y-axis
#' @param title Title of plot.
#' @param asPoints Whether to plot points using individual symbols or counting bins
#' @param nbins Number of discretization bins 
#' @param cmap Color map used to plot the points by color. Can be specified by a name (cf. \code{\link{printAllPalettes}} to display available choices), or by a vector of color names which will be interpolated to create a palette.
#' @param regrLine Whether to add a regression line to the plot
#' @param regrLineColor,regrLineType,regrLineWidth Color, type and width of the regression line
#' @param eqLine Whether to add a regression line x=y to the plot
#' @param eqLineColor,eqLineType,eqLineWidth Color, type and width of the  line x=y
#' @param verbose Whether to print the regression results (when `regrLine=TRUE`)
#' 
#' @return A \pkg{ggplot2} object containing the plot. If stored in a variable, use the function \code{print} to display the plot.
#' 
#' @export
#'
#' 
plot_scatter<-function (db,vnamex,vnamey,
                        xlab=NULL,ylab=NULL,title=NULL,
                        asPoints=FALSE,nbins=25,
                        cmap="viridis",
                        regrLine=FALSE, regrLineColor = "blue", regrLineType = "solid",regrLineWidth = 1.25,
                        eqLine=FALSE, eqLineColor = "red", eqLineType = "dashed", eqLineWidth = 1.25,
                        verbose=TRUE){
  p = plot.init()
  
  res = correlationPairs(db, db, vnamex, vnamey)
  x = db$getValuesByNames(res[[1]], vnamex)
  y = db$getValuesByNames(res[[2]], vnamey)
  
  if (asPoints)
    p = p + plot.XY(x, y, flagLine=FALSE, flagPoint=TRUE)
  else
    p = p + geom_bin_2d(aes())
    p = p + plot.hist2d(x, y,bins=nbins)
    p = p +.defineFill(cmap)

  if (eqLine)
  {
      p=p+geom_abline(intercept=0,slope=1,
                      linetype =eqLineType, color = eqLineColor,linewidth=eqLineWidth)
    }
  if (regrLine)
  {
    regr = regression(db, vnamey, vnamex, flagCst=TRUE)
    if (regr$getNvar() > 0)
    {
      a = regr$getCoeff(0)
      b = regr$getCoeff(1)
      r2=1-sum((a+b*db[vnamex]-db[vnamey])**2,na.rm=T)/sum((mean(db[vnamey],na.rm=T)-db[vnamey])**2,na.rm=T)
      p=p+geom_abline(intercept=a,slope=b,
                      linetype = regrLineType, color = regrLineColor,linewidth=regrLineWidth)
      if(verbose){
        message(paste0("Regression results: Slope=",b,"; Intercept=",a,"; R2=",r2))
      }
    }
  }
  
  p = p + plot.decoration(title=title, 
                          xlab=xlab, ylab=ylab)
  return(p)
}

