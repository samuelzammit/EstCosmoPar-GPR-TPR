# load hetGP package
library(hetGP)

#### FUNCTION DEFINITION ####
# function for fitting and plotting
hetGPFitPlot <- function(gpr=TRUE, tpr=TRUE, regHom=TRUE, regHet=TRUE, covname="Matern5_2", dataname) {
  
  # plotting function
  plotRegression <- function(Xgrid, homfit, hetfit, hompred, hetpred, methodname, dataname, dataname2, xmax, ymin, ymax, ylabel="H(z)") {
    
    # set working directory for saving plots
    setwd("G:\\My Drive\\MSc\\SOR5200 - Dissertation\\GaPP_27\\Application\\5_hetGP\\files")
    
    # set plot colours, line width, line type
    colOption = 'red'
    lwdOption = 2
    ltyOption = 2
    
    # if there are any 'open' plots, close them down
    while(dev.cur() > 1) dev.off()
    
    # if regHom and regHet are both FALSE, then stop in error
    if(!regHom && !regHet) stop("Error: at least one of regHom and regHet must be TRUE")
    
    # plot and save homoscedastic fit and p/m 1 s.d.
    if(regHom) {
      
      if(methodname == "GPR") {
        pdf(paste0("regHom_GPR_", dataname, "_", covname, ".pdf"))
      } else if(methodname == "TPR") {
        pdf(paste0("regHom_TPR_", dataname, "_", covname, ".pdf"))
      } else {
        stop("Invalid method name! Should be one of 'GPR' or 'TPR'.")
      }
      
      plot(dataname2, main = paste(methodname, dataname, sep=", "), ylim = c(ymin, ymax), xlim = c(0, xmax), ylab = ylabel, xlab = "z")
      lines(Xgrid, hompred$mean, col = colOption, lwd = lwdOption, lty = ltyOption)
      lines(Xgrid, qnorm(0.32, hompred$mean, sqrt(hompred$sd2 + hompred$nugs)), col = colOption)
      lines(Xgrid, qnorm(0.68, hompred$mean, sqrt(hompred$sd2 + hompred$nugs)), col = colOption)
      dev.off()
            
    }
    
    # plot and save heteroscedastic fit and p/m 1 s.d.
    if(regHet) {
      
      if(methodname == "GPR") {
        pdf(paste0("regHet_GPR_", dataname, "_", covname, ".pdf"))
      } else if(methodname == "TPR") {
        pdf(paste0("regHet_TPR_", dataname, "_", covname, ".pdf"))
      } else {
        stop("Invalid method name! Should be one of 'GPR' or 'TPR'.")
      }
      
      plot(dataname2, main = paste(methodname, dataname, sep=", "), ylim = c(ymin, ymax), xlim = c(0, xmax), ylab = ylabel, xlab = "z")
      lines(Xgrid, hetpred$mean, col = colOption, lwd = lwdOption, lty = ltyOption)
      lines(Xgrid, qnorm(0.32, hompred$mean, sqrt(hetpred$sd2 + hetpred$nugs)), col = colOption, lty = ltyOption)
      lines(Xgrid, qnorm(0.68, hompred$mean, sqrt(hetpred$sd2 + hetpred$nugs)), col = colOption, lty = ltyOption)
      # empSd <- sapply(find_reps(dataname2$z, dataname2$H)$Zlist, sd)
      # points(hetfit$X0, hetfit$Z0, pch = 20)
      # arrows(x0 = hetfit$X0, y0 = qnorm(0.05, hetfit$Z0, empSd), y1 = qnorm(0.95, hetfit$Z0, empSd), code = 3, angle = 90, length = 0.01)
      dev.off()
      
    }
    
  }
  
  # for measuring runtime
  start = Sys.time()
  
  # load data
  wd <- "G:\\My Drive\\MSc\\SOR5200 - Dissertation\\GaPP_27\\Application\\data"
  if(dataname == "FullPantheon_Data") {
    dataset <- read.csv(paste0(wd, "\\FullPantheon_Data.txt"), sep="\t", header=FALSE)
  } else {
    dataset <- read.csv(paste0(wd, "\\Hdata_", dataname, ".txt"), sep=" ", header=FALSE)
  }
  names(dataset) <- c("z", "H", "sigmaH")
  
  # discard sigmaH
  dataset <- dataset[, 1:2]  
  
  # homoscedastic and heteroscedastic GP and TP fits on data with kernel specified by covname
  # available kernels: Gaussian, Matern3_2, Matern5_2
  # we are not including the observation variance (i.e. third column in the data) here
  # we are also not including the covariance matrix of the data
  # the idea (for heteroscedastic regression) is that the variance is increased where the readings are more sparse
  # (by doing MLE with additional parameter for the noise variance instead of assuming iid)
  if(regHom && gpr) homGPfit <- mleHomGP(dataset$z, dataset$H, covtype = covname)
  if(regHet && gpr) hetGPfit <- mleHetGP(dataset$z, dataset$H, covtype = covname)
  if(regHom && tpr) homTPfit <- mleHomTP(dataset$z, dataset$H, covtype = covname)
  if(regHet && tpr) hetTPfit <- mleHetTP(dataset$z, dataset$H, covtype = covname)
  
  # prediction
  Xgrid <- matrix(seq(0, 2, length = 301), ncol = 1)
  if(regHom && gpr) homGPpred <- predict(x = Xgrid, object = homGPfit)
  if(regHet && gpr) hetGPpred <- predict(x = Xgrid, object = hetGPfit)
  if(regHom && tpr) homTPpred <- predict(x = Xgrid, object = homTPfit)
  if(regHet && tpr) hetTPpred <- predict(x = Xgrid, object = hetTPfit)
  
  # plotting
  if(gpr) plotRegression(Xgrid, homGPfit, hetGPfit, homGPpred, hetGPpred, "GPR", dataname, dataset, 2, 60, 250)
  if(tpr) plotRegression(Xgrid, homTPfit, hetTPfit, homTPpred, hetTPpred, "TPR", dataname, dataset, 2, 60, 250)
  
  # print parameter estimates
  if(regHom && gpr) cat(paste0(dataname, ", Homoscedastic GPR: ", round(homGPpred$mean[1], 3), " p/m ", round(sqrt(homGPpred$sd2[1]), 3), "\n"))
  if(regHet && gpr) cat(paste0(dataname, ", Heteroscedastic GPR: ", round(hetGPpred$mean[1], 3), " p/m ", round(sqrt(hetGPpred$sd2[1]), 3), "\n"))
  if(regHom && tpr) cat(paste0(dataname, ", Homoscedastic TPR: ", round(homTPpred$mean[1], 3), " p/m ", round(sqrt(homTPpred$sd2[1]), 3), "\n"))
  if(regHet && tpr) cat(paste0(dataname, ", Heteroscedastic TPR: ", round(hetTPpred$mean[1], 3), " p/m ", round(sqrt(hetTPpred$sd2[1]), 3), "\n"))

  # print "done" message
  cat(paste0(covname, ", ", dataname, ": DONE\n"))

}

#### FUNCTION CALLS ####
hetGPFitPlot(dataname="CC")
hetGPFitPlot(dataname="CC+SN")
hetGPFitPlot(dataname="CC+SN+BAO")
hetGPFitPlot(dataname="FullPantheon_Data")