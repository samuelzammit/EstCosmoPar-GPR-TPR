# load hetGP package
library(hetGP)

#### MAIN FUNCTION DEFINITION ####
runHetGP <- function(type, dataname, covname, priorname, priorvalue, priorsd, regHom=TRUE, regHet=TRUE) {
  
  #### SUB-FUNCTION DEFINITION ####
  # plotting function
  plotRegression <- function(Xgrid, type, homfit, hetfit, hompred, hetpred, dataname, dataname2, xmax, ymin, ymax, ylabel="H(z)") {
    
    # set plot colours, line width, line type
    colOption = 'red'
    lwdOption = 2
    ltyOption = 2
    
    # if there are any 'open' plots, close them down
    while(dev.cur() > 1) dev.off()
    
    # if regHom and regHet are both FALSE, then stop in error
    if(!regHom && !regHet) stop("Error: at least one of regHom and regHet must be TRUE")
    
    # if regHom, plot and save homoscedastic fit and p/m 1 s.d.
    if(regHom) {
      
      if(priorname == "NoPrior") {
        if(type == "GPR") {
          pdf(paste0("regHom_GPR_", dataname, "_", covname, "_NoPrior.pdf"))
        } else if(type == "TPR") {
          pdf(paste0("regHom_TPR_", dataname, "_", covname, "_NoPrior.pdf"))
        } else {
          stop("Invalid type in function plotRegression! Should be one of 'GPR' or 'TPR'.")
        }
      } else {
        if(type == "GPR") {
          pdf(paste0("regHom_GPR_", dataname, "_", covname, "_", priorname, ".pdf"))
        } else if(type == "TPR") {
          pdf(paste0("regHom_TPR_", dataname, "_", covname, "_", priorname, ".pdf"))
        } else {
          stop("Invalid type in function plotRegression! Should be one of 'GPR' or 'TPR'.")
        }
      }
      
      if(priorname == "NoPrior") {
        ptitle = paste(type, dataname, "No prior", sep=", ")
      } else {
        ptitle = paste0(type, ", ", dataname, ", ", priorname, " prior")
      } 
      plot(dataname2, main = ptitle, ylim = c(ymin, ymax), xlim = c(0, xmax), ylab = ylabel, xlab = "z")
      lines(Xgrid, hompred$mean, col = colOption, lwd = lwdOption, lty = ltyOption)
      lines(Xgrid, qnorm(0.32, hompred$mean, sqrt(hompred$sd2 + hompred$nugs)), col = colOption)
      lines(Xgrid, qnorm(0.68, hompred$mean, sqrt(hompred$sd2 + hompred$nugs)), col = colOption)
      dev.off()
      
    }
    
    # if regHet, plot and save heteroscedastic fit and p/m 1 s.d.
    if(regHet) {
      
      if(priorname == "NoPrior") {
        if(type == "GPR") {
          pdf(paste0("regHet_GPR_", dataname, "_", covname, "_NoPrior.pdf"))
        } else if(type == "TPR") {
          pdf(paste0("regHet_TPR_", dataname, "_", covname, "_NoPrior.pdf"))
        } else {
          stop("Invalid type in function plotRegression! Should be one of 'GPR' or 'TPR'.")
        }
      } else {
        if(type == "GPR") {
          pdf(paste0("regHet_GPR_", dataname, "_", covname, "_", priorname, ".pdf"))
        } else if(type == "TPR") {
          pdf(paste0("regHet_TPR_", dataname, "_", covname, "_", priorname, ".pdf"))
        } else {
          stop("Invalid type in function plotRegression! Should be one of 'GPR' or 'TPR'.")
        }
      }
      
      if(priorname == "NoPrior") {
        ptitle = paste(type, dataname, "No prior", sep=", ")
      } else {
        ptitle = paste0(type, ", ", dataname, ", ", priorname, " prior")
      } 
      plot(dataname2, main = ptitle, ylim = c(ymin, ymax), xlim = c(0, xmax), ylab = ylabel, xlab = "z")
      lines(Xgrid, hetpred$mean, col = colOption, lwd = lwdOption, lty = ltyOption)
      lines(Xgrid, qnorm(0.32, hetpred$mean, sqrt(hetpred$sd2 + hetpred$nugs)), col = colOption)
      lines(Xgrid, qnorm(0.68, hetpred$mean, sqrt(hetpred$sd2 + hetpred$nugs)), col = colOption)
      dev.off()
      
    }
    
  }
  
  # function to calculate "sigma distance" between H_0 estimate and prespecified prior
  calcDistance <- function(H0est, H0estpm, priorval, priorvalpm) {
    dist <- (H0est - priorval) / sqrt(H0estpm^2 + priorvalpm^2)
    rdist <- round(dist, 4)
    return(rdist)
  }
  
  # load data
  if(dataname == "FullPantheon") {
    dataset <- read.csv(paste0(wd, "\\FullPantheon_Data.txt"), sep="\t", header=FALSE)
  } else {
    dataset <- read.csv(paste0(wd, "\\Hdata_", dataname, ".txt"), sep=" ", header=FALSE)
  }
  
  names(dataset) <- c("z", "H", "sigmaH")
  
  # discard sigmaH
  dataset <- dataset[, 1:2]
  
  # if prior is specified then add it (as an artificial point at z=0) to the data
  if(!is.na(priorvalue)) dataset <- rbind(c(0, priorvalue), dataset)
  
  # define x- and y-axis ranges
  xmin <- 0
  if(dataname == "FullPantheon") {
    xmax <- 1.5
    ymin <- 10
    ymax <- 100
  } else {
    xmax <- 2
    ymin <- 60
    ymax <- 250
  }
  
  # runGPTP(type, dataname, covname, xmin, xmax, ymin, ymax)
  # homoscedastic and heteroscedastic GP and TP fits on data with kernel specified by covname
  # available kernels: Gaussian, Matern3_2, Matern5_2
  # we are not including the observation variance (i.e. third column in the data) here
  # we are also not including the covariance matrix of the data
  # the idea (for heteroscedastic regression) is that the variance is increased where the readings are more sparse
  # (by doing MLE with additional parameter for the noise variance instead of assuming iid)
  if(type == "GPR") {
    homfit <- mleHomGP(dataset$z, dataset$H, covtype = covname)
    hetfit <- mleHetGP(dataset$z, dataset$H, covtype = covname)
  } else if(type == "TPR") {
    homfit <- mleHomTP(dataset$z, dataset$H, covtype = covname)
    hetfit <- mleHetTP(dataset$z, dataset$H, covtype = covname)
  } else {
    stop("Invalid type parameter in function runGPTP")
  }
  
  # prediction
  Xgrid <- matrix(seq(xmin, xmax, length = 301), ncol = 1)
  hompred <- predict(x = Xgrid, object = homfit)
  hetpred <- predict(x = Xgrid, object = hetfit)
  
  # plotting
  plotRegression(Xgrid, type, homfit, hetfit, hompred, hetpred, dataname, dataset, xmax, ymin, ymax)
  
  # print parameter estimates and distance calculations (distance in sigma-units)
  if(priorname == "NoPrior") {
    cat(paste0(dataname, ", Homoscedastic ", type, ": ", round(hompred$mean[1], 3), " p/m ", round(sqrt(hompred$sd2[1]), 3), "\n"))
    for(j in 2:length(prinames)) cat(paste0("Distance to ", prinames[j], ": ", calcDistance(hompred$mean[1], sqrt(hompred$sd2[1]), privalues[j], prisds[j])), "\n")
    cat("\n")
    
    cat(paste0(dataname, ", Heteroscedastic ", type, ": ", round(hetpred$mean[1], 3), " p/m ", round(sqrt(hetpred$sd2[1]), 3), "\n"))
    for(j in 2:length(prinames)) cat(paste0("Distance to ", prinames[j], ": ", calcDistance(hetpred$mean[1], sqrt(hetpred$sd2[1]), privalues[j], prisds[j])), "\n")
    cat("\n")
    
  } else {
    
    cat(paste0(dataname, ", Homoscedastic ", type, ", ", priorname, " prior: ", round(hompred$mean[1], 3), " p/m ", round(sqrt(hompred$sd2[1]), 3), "\n"))
    for(j in 2:length(prinames)) cat(paste0("Distance to ", prinames[j], ": ", calcDistance(hompred$mean[1], sqrt(hompred$sd2[1]), privalues[j], prisds[j])), "\n")
    cat("\n")
    
    cat(paste0(dataname, ", Heteroscedastic ", type, ", ", priorname, " prior: ", round(hetpred$mean[1], 3), " p/m ", round(sqrt(hetpred$sd2[1]), 3), "\n"))
    for(j in 2:length(prinames)) cat(paste0("Distance to ", prinames[j], ": ", calcDistance(hetpred$mean[1], sqrt(hetpred$sd2[1]), privalues[j], prisds[j])), "\n")
    cat("\n")
    
  }
  
  # print "done" message
  if(priorname == "NoPrior") {
    cat(paste0(covname, ", ", dataname, ", ", type, ": DONE\n"))
  } else {
    cat(paste0(covname, ", ", dataname, ", ", type, ", ", priorname, ": DONE\n"))
  }
  
}

#### RUNNING THE PROGRAM ####

# for measuring runtime
start = Sys.time()

# define list of methods, datasets, kernels, priors
methods <- c("GPR", "TPR")
datnames <- c("CC", "CC+SN", "CC+SN+BAO", "FullPantheon")
kernames <- c("Gaussian", "Matern3_2", "Matern5_2")
prinames <- c("NoPrior", "Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES")
privalues <- c(NA, 74.22, 69.8, 73.3, 75.35, 67.4, 67.4)
prisds <- c(NA, 1.82, 1.9, 1.75, 1.68, 0.5, 1.15)

# directory containing the data
wd <- "G:\\My Drive\\MSc\\SOR5200 - Dissertation\\GaPP_27\\Application\\data"

# set working directory for saving plots and parameter estimates
setwd("G:\\My Drive\\MSc\\SOR5200 - Dissertation\\GaPP_27\\Application\\5_hetGP\\files")

# start text output
sink("output.txt")

# function calls
for(i in 1:length(methods)) {

  for(j in 1:length(datnames)) {
    
    for(k in 1:length(kernames)) {
      
      for(l in 1:length(prinames)) {
        
        message(paste(methods[i], datnames[j], kernames[k], prinames[l], sep=", "))
        
        # "if" condition is there so that Pantheon data is only run for priorless case
        # therefore, if either (i) data is not FullPantheon and/or (ii) priorless case then run
        if(datnames[j] != "FullPantheon" || prinames[l] == "NoPrior") {
          runHetGP(methods[i], datnames[j], kernames[k], prinames[l], privalues[l], prisds[l])
        } else {
          message("Skipped")
        }
        
        message("\n")
        
      }
      
    }
    
  }
  
}

# end text output
while(sink.number() >= 1) sink()

# reset working directory
setwd("~/")

message("Done")