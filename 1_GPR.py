# import required packages/modules
import os
import sys
import warnings
import numpy as np
from gapp import covariance as cov
from gapp import dgp

sys.setrecursionlimit(100000)
warnings.simplefilter(action='ignore', category=FutureWarning)

# set working directory
os.chdir("~/GaPP_27/Application/1_dgp_new/files/")


# dgphubble performs GPR reconstruction of the H(z)-against-z curve given a specified covariance function/prior/dataset
def dgphubble(prior=False, priorval=None, priorsdev=None, saveoutput=True, plotoutput=True, dataname, priorname, covfn):
    if __name__ == "__main__":

        # load Hubble data
        (z, H, sigmaH) = np.loadtxt("~/GaPP_27/Application/data/Hdata_" + dataname + ".txt", unpack=True)

        # prepend priors
        if prior:
            z = np.insert(z, 0, 0, axis=0)
            H = np.insert(H, 0, priorval, axis=0)
            sigmaH = np.insert(sigmaH, 0, priorsdev, axis=0)
        else:
            priorval = None
            priorsdev = None

        # nstar points of the function will be reconstructed between zmin and zmax
        zmin = 0.0
        zmax = 2.5 if dataname == "CC+SN+BAO" else 2.0
        nstar = 200

        # determine name of covariance function used
        covname = {
            cov.SquaredExponential: "SquEx",
            cov.DoubleSquaredExponential: "DoubleSquEx",
            cov.RationalQuadratic: "RatQuadratic",
            cov.Matern32: "Matern32",
            cov.Matern52: "Matern52",
            cov.Matern72: "Matern72",
            cov.Matern92: "Matern92",
            cov.Cauchy: "Cauchy"
        }.get(covfn, "ERROR: invalid covariance function")

        # determine length of parameter vector theta and set initial values
        # initial values are picked at random from [0,1000)
        x1 = 1000*np.random.random()
        x2 = 1000*np.random.random()
        x3 = 1000*np.random.random()
        x4 = 1000*np.random.random()

        initheta = {
            cov.SquaredExponential: [x1, x2],                  # [sigma_f, l]
            cov.DoubleSquaredExponential: [x1, x2, x3, x4],    # [sigma_f1, l1, sigma_f2, l2]
            cov.RationalQuadratic: [x1, x2, x3],               # [sigma_f, l, alpha]
            cov.Matern32: [x1, x2],                            # [sigma_f, l]
            cov.Matern52: [x1, x2],                            # [sigma_f, l]
            cov.Matern72: [x1, x2],                            # [sigma_f, l]
            cov.Matern92: [x1, x2],                            # [sigma_f, l]
            cov.Cauchy: [x1, x2]                               # [sigma_f, l]
        }.get(covfn, "ERROR: invalid covariance function")

        # print information on dataset, covariance function, prior
        print("Dataset: " + dataname)

        print("Covariance function: " + covname)

        if priorval is not None and priorsdev is not None:
            print("Prior: " + str(priorval) + " +/- " + str(priorsdev) + " (" + priorname + ")")
        else:
            print("No prior")

        # initialisation of GP
        g = dgp.DGaussianProcess(z, H, sigmaH, cXstar=(zmin, zmax, nstar), covfunction=covfn)

        # training of the hyperparameters and reconstruction of the function
		# GaPP also allows for reconstruction of derivatives, but this is not used
        (rec, theta) = g.gp(theta=initheta)

        # save the output
        if saveoutput:
            np.savetxt("H_" + dataname + "_" + covname + "_" + priorname + ".txt", rec)

        # plot the output
        if plotoutput:
            import plottingfunction
            plotname = "plot_" + dataname + "_" + covname + "_" + priorname
            plottitle = dataname + "/" + covname + "/" + priorname
            print("Plotting " + plottitle)
            plottingfunction.plot(z, H, sigmaH, rec, zmin, zmax, plotname, plottitle)

        # return loglikelihood for each model
        return g.log_likelihood()


# define lists of data sources, covariance functions, priors (and their values)
datas = ["CC", "CC+SN", "CC+SN+BAO"]
covfunctions = [cov.SquaredExponential, cov.DoubleSquaredExponential, cov.RationalQuadratic,
                cov.Matern32, cov.Matern52, cov.Matern72, cov.Matern92, cov.Cauchy]
priorvals = [74.22, 69.8, 73.3, 75.35, 67.4, 67.4]
priorsdevs = [1.82, 1.9, 1.7, 1.68, 0.5, 1.1]
priornames = ["Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES"]

# store log-likelihood for each model
# len(priornames)+1 to include also priorless case
logl = np.zeros((len(datas), len(covfunctions), len(priornames)+1))

# loop over all datasets/cov. fn./priors
for i in range(len(datas)):
    for j in range(len(covfunctions)):

        # no prior
        print
        print(i, j, 0)
        logl[i, j, 0] = dgphubble(dataname=datas[i], priorname="NoPrior", covfn=covfunctions[j])

        # for each prior:
        for k in range(len(priornames)):
            print
            print(i, j, k+1)
            logl[i, j, k+1] = dgphubble(True, priorvals[k], priorsdevs[k], dataname=datas[i], priorname=priornames[k], covfn=covfunctions[j])
            logl[i, j, k+1] = round(logl[i, j, k+1], 4)

print(logl)
