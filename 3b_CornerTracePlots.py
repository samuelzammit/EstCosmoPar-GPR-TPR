# import required packages/modules
import corner
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def traceandcorner(dataname, covname, priorname, nwalker, niter):

    os.chdir("~/GaPP_27/Application/3_mcmcdgp_new/files/thetasample")
    data = pd.read_csv("thetasample_" + dataname + "_" + covname + "_" + priorname + ".txt", sep=" ", header=None)

    os.chdir("~/GaPP_27/Application/3_mcmcdgp_new/files/logl")
    logl = pd.read_csv("logl_" + dataname + "_" + covname + "_" + priorname + ".txt", sep=" ", header=None)

    # determine list of parameters for trace/corner plots
    data.columns = {
        "SquEx": ['sigma_f', 'l'],
        "DoubleSquEx": ['sigma_{f1}', 'l_1', 'sigma_{f2}', 'l_2'],
        "RatQuadratic": ['sigma_f', 'l', 'alpha'],
        "Matern32": ['sigma_f', 'l'],
        "Matern52": ['sigma_f', 'l'],
        "Matern72": ['sigma_f', 'l'],
        "Matern92": ['sigma_f', 'l'],
        "Cauchy": ['sigma_f', 'l']
    }.get(covname, "ERROR: invalid covariance function")

    npar = data.shape[1]  # number of parameters (= number of columns in thetasample)

    # check nwalker * niter = number of rows of thetasample
    if not (nwalker * niter == data.shape[0]):
        ValueError("Warning: nwalker * niter not equal to number of rows of thetasample")

    # check nwalker * niter = number of rows of thetasample
    if not (nwalker * niter == logl.shape[0]):
        ValueError("Warning: nwalker * niter not equal to number of rows of loglikelihood text file")

    dataijavg = np.zeros((niter, npar))
    loglijavg = np.zeros((niter, 1))

    for iters in range(niter):

        start = nwalker * iters
        end = nwalker * iters + nwalker

        # calculate average theta at each iteration
        for par in range(npar):
            dataijavg[iters, par] = np.mean(data.iloc[start:end, par])

        # calculate average loglikelihood at each iteration
        loglijavg[iters] = logl.iloc[iters]

    plotname = "plot_" + dataname + "_" + covname + "_" + priorname

    # trace plot (to loop over par = 0, 1, ...)
    os.chdir("../trace")
    for par in range(npar):
        plt.figure()
        plt.plot(dataijavg[:, par])
        plt.xlabel("Iteration")
        plt.ylabel(data.columns[par])
        plt.savefig("trace_" + plotname + "_" + data.columns[par] + ".pdf")
        plt.close()

    # corner plot
    os.chdir("../corner")
    corner.corner(dataijavg, labels=data.columns)
    plt.savefig("corner_" + plotname + ".pdf")
    plt.close()

    # loglikelihood plot
    os.chdir("../logl")
    plt.figure()
    plt.plot(loglijavg)
    plt.savefig("logl_" + plotname + ".pdf")
    plt.close()

    # go back to folder containing thetasample
    os.chdir("../thetasample/")


# define data sources, covariance functions, prior
datas = ["CC", "CC+SN", "CC+SN+BAO"]
covnames = ["SquEx", "DoubleSquEx", "RatQuadratic", "Matern32", "Matern52", "Matern72", "Matern92", "Cauchy"]
priornames = ["NoPrior", "Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES"]

# loop over all datasets/cov. fn./priors
for i in range(len(datas)):
    for j in range(len(covnames)):
        for k in range(len(priornames)):

            print
            print(i, j, k)
            traceandcorner(datas[i], covnames[j], priornames[k], nwalker=100, niter=100000)
