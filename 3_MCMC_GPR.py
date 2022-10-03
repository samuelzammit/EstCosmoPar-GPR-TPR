# import required packages/modules
import os
import sys
import warnings
import numpy as np
from gapp import covariance as cov
from gapp import mcmcdgp

sys.setrecursionlimit(100000)

warnings.filterwarnings('ignore', message="numpy.dtype size changed")
warnings.filterwarnings('ignore', message="numpy.ufunc size changed")

# set working directory
os.chdir("~/GaPP_27/Application/3_mcmcdgp_new/files")


# mcmcdgphubble performs MCMC GP reconstruction of the H(z)-against-z curve given a specified covariance function/prior/dataset
def mcmcdgphubble(prior=False, priorval=None, priorsdev=None, dataname="CC+SN+BAO", priorname="NoPrior",
                  covfn=None, nwalker=100, niter=5000, plotoutput=True, nthread=1):
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
        nstar = 300

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

        # print information on dataset, covariance function, prior
        print("Dataset: " + dataname)

        print("Covariance function: " + covname)

        if priorval is not None and priorsdev is not None:
            print("Prior: " + str(priorval) + " +/- " + str(priorsdev) + " (" + priorname + ")")
        else:
            print("No prior")

        # initialise theta as (nwalker, npar) dimensional and add random noise
        # determine number of hyperparamters of kernel
        npar = {
            cov.SquaredExponential: 2,
            cov.DoubleSquaredExponential: 4,
            cov.RationalQuadratic: 3,
            cov.Matern32: 2,
            cov.Matern52: 2,
            cov.Matern72: 2,
            cov.Matern92: 2,
            cov.Cauchy: 2
        }.get(covfn, "ERROR: invalid covariance function")

        # initialise theta as (nwalker, npar) dimensional and add random noise
        theta0 = np.random.normal(10, 1, (nwalker, npar))
        theta0 = theta0 + np.random.normal(0, 0.5, [nwalker, npar])

        # initialisation of MCMC GP
        mcmcg = mcmcdgp.MCMCDGaussianProcess(z, H, sigmaH, theta0, covfunction=covfn,
                                             Niter=niter, cXstar=(zmin, zmax, nstar),
                                             threads=nthread, reclist=derivlist)

        # training of the hyperparameters
        # this outputs thetasample, which can then be used for reconstruction
        thetasample = mcmcg.mcmcdgp(dataname, covname, priorname)

        # get thetasample as at final iteration
        opttheta = np.zeros((npar,))
        start = nwalker * (niter - 1)
        end = nwalker * niter
        for par in range(npar):
            print(thetasample[start:end, par])
            3)
            opttheta[par] = np.mean(thetasample[start:end, par])

            print(opttheta)

            # reconstruction (using optimised hyperparameter values from mcmcdgp)
            # thetatrain = False: no need to train theta again
            (rec, theta) = mcmcg.gp(theta=opttheta, thetatrain='False')

            np.savetxt("H_" + dataname + "_" + covname + "_" + priorname + ".txt", rec)

            # plot the output
            if plotoutput:
                import plottingfunction
            plotname = "plot_" + dataname + "_" + covname + "_" + priorname
            plottitle = dataname + "/" + covname + "/" + priorname
            print("Plotting " + plottitle)
            plottingfunction.plot(z, H, sigmaH, rec, zmin, zmax, plotname, plottitle)

            # return loglikelihood for each model
        return mcmcg.log_likelihood()


# print output to file
sys.stdout = open('output.txt', 'w')

# define data sources, covariance functions, prior
datas = ["CC", "CC+SN", "CC+SN+BAO"]
covfunctions = [cov.SquaredExponential, cov.DoubleSquaredExponential, cov.RationalQuadratic,
                cov.Matern32, cov.Matern52, cov.Matern72, cov.Matern92, cov.Cauchy]
priorvals = [74.22, 69.8, 73.3, 75.35, 67.4, 67.4]
priorsdevs = [1.82, 1.9, 1.7, 1.68, 0.5, 1.1]
priornames = ["Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES"]

# number of walkers and iterations
nwalk = 100
nit = 100000

# to store log-likelihood for each model
# len(priornames)+1 to include also priorless case
logl = np.zeros((len(datas), len(covfunctions), len(priornames) + 1))

# loop over all datasets/cov. fn./priors
for i in range(len(datas)):

    for j in range(len(covfunctions)):

        # no prior
        print
        print(i, j, 0)
        logl[i, j, 0] = mcmcdgphubble(dataname=datas[i], covfn=covfunctions[j], nwalker=nwalk, niter=nit)

        for k in range(len(priornames)):
            print
            print(i, j, k + 1)
            logl[i, j, k + 1] = mcmcdgphubble(True, priorvals[k], priorsdevs[k], datas[i], priornames[k], covfunctions[j], nwalker=nwalk, niter=nit)

print(logl)
sys.stdout.close()
