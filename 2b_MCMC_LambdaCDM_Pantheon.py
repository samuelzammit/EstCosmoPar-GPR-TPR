# import required packages
import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate
from scipy.linalg import cholesky, solve_triangular

os.chdir("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/2_mcmc_emcee_LambdaCDM/files/new_emcee_LambdaCDM_Pantheon")


# 20k, 200
# emcee_LambdaCDM carries out MCMC inference using the Lambda CDM model with parameters H0 and Omega_{M0}
def emcee_LambdaCDM(prior=False, priorval=None, priorsdev=None, dataname="CC+SN+BAO", priorname="NoPrior", niter=10000,
                    nwalker=6, plotoutput=True, saveoutput=True):

    if __name__ == "__main__":

        # load Hubble data
        (zCC, HCC, sigmaHCC) = np.loadtxt("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/data/Hdata_CC.txt", unpack=True)

        (zSN, muSN, sigmamuSN) = np.loadtxt("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/data/FullPantheon_Data.txt", unpack=True)
        covSN = np.loadtxt("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/data/FullPantheon_CorrMatrix.txt", unpack=True)

        (zBAO, HBAO, sigmaHBAO) = np.loadtxt("G:/My Drive/MSc/SOR5200 - Dissertation/GaPP_27/Application/data/Hdata_BAO.txt", unpack=True)

        # prepend priors (to CC dataset, but not strictly correct)
        if prior:
            zCC = np.insert(zCC, 0, 0, axis=0)
            HCC = np.insert(HCC, 0, priorval, axis=0)
            sigmaHCC = np.insert(sigmaHCC, 0, priorsdev, axis=0)
        else:
            priorval = None
            priorsdev = None

        (z, H, sigmaH) = (zCC, HCC, sigmaHCC)

        if dataname == "CC+SN" or dataname == "CC+SN+BAO":
            (z, H, sigmaH) = (np.append(z, zSN), np.append(H, muSN), np.append(sigmaH, sigmamuSN))

            if dataname == "CC+SN+BAO":
                (z, H, sigmaH) = (np.append(z, zBAO), np.append(H, HBAO), np.append(sigmaH, sigmaHBAO))

        # function definition
        # prior (mostly uninformative)
        def log_prior(theta):
            H0, Omega_M0, M = theta
            if 50 < H0 < 100 and 0.1 < Omega_M0 < 0.5 and -50 < M < 50:
                return 0.0
            return -np.inf

        # likelihood (CC, SN, BAO components)
        # CC: "usual" Chi2
        def log_likelihood_CC(theta):
            H0, Omega_M0, _ = theta
            Hpred = H0 * np.sqrt(Omega_M0 * (1 + zCC) ** 3 + (1 - Omega_M0))
            return -0.5 * np.sum((HCC - Hpred) ** 2 / sigmaHCC ** 2)

        # SN: see equation (18) of arXiv:2108.03853
        def log_likelihood_SN(theta):
            H0, Omega_M0, M = theta

            # recover the distance moduli from CLASS
            moduli = np.empty((zSN.size,))
            
            # luminosity distance
            # calculated as a numerical integral, see equation (17) of arXiv:2108.03853

            # 1. calculate luminosity distance at specific redshift values
            # i.e. working out the integral from z=0 to zPos (z_i)
            def dL(zPos, H0, Omega_M0):
                def dLInt(zPosition):
                    H_Int = H0 * np.sqrt(Omega_M0 * (1 + zPosition) ** 3 + (1 - Omega_M0))
                    return 1 / H_Int

                c = 299792458  # speed of light; constant

                return c * (1 + zPos) * integrate.romberg(dLInt, 0, zPos)

            # 2. build luminosity distance interpolator
            def dL_interpret(H0, Omega_M0):
                dL_interp = np.zeros(len(zSN))

                for index, z in enumerate(zSN):
                    dL_temp = dL(z, H0, Omega_M0)
                    if not np.isnan(dL_temp):
                        dL_interp[index] = dL_temp
                    else:
                        return -np.inf

                return interpolate.interp1d(zSN, dL_interp)  # , kind='quadratic')

            # setup interpreter for each walker chain step
            dL_interp = dL_interpret(H0, Omega_M0)
            dLum = dL_interp(zSN)
            moduli = 5 * np.log10(dLum) + 25
    
            # compute the covariance matrix
            # cov = ne.evaluate("covSN")
    
            # compute the residuals (estimate of distance moduli - exact moduli)
            residuals = np.empty((zSN.size,))
            
            # compute the approximate moduli
            residuals = muSN - M
            
            # remove from the approximate moduli the one computed from CLASS
            residuals -= moduli
    
            # update the diagonal terms of the covariance matrix with the statistical error
            cov = covSN + np.diag(sigmamuSN ** 2)
    
            # compute the Cholesky decomposition of the covariance matrix
            cov = cholesky(cov, lower=True, overwrite_a=True)
    
            # solve the triangular system
            residuals = solve_triangular(cov, residuals, lower=True, check_finite=False)
    
            # return the likelihood (= -0.5 * chisq)
            return -0.5 * (residuals ** 2).sum()

        # BAO: "usual" Chi2
        def log_likelihood_BAO(theta):
            H0, Omega_M0, _ = theta
            Hpred = H0 * np.sqrt(Omega_M0 * (1 + zBAO) ** 3 + (1 - Omega_M0))
            return -0.5 * np.sum((HBAO - Hpred) ** 2 / sigmaHBAO ** 2)

        # posterior
        def log_probability(theta, z, H, sigmaH):
            logpri = log_prior(theta)
            if not np.isfinite(logpri):
                return -np.inf
            if dataname == "CC":
                logpost = log_likelihood_CC(theta)
            elif dataname == "CC+SN":
                logpost = logpri + log_likelihood_CC(theta) + log_likelihood_SN(theta)
            elif dataname == "CC+SN+BAO":
                logpost = logpri + log_likelihood_CC(theta) + log_likelihood_SN(theta) + log_likelihood_BAO(theta)
            return logpost

    print("\nDataset: " + dataname)

    # print information on dataset, covariance function, prior
    if priorval is not None and priorsdev is not None:
        print("Prior: " + str(priorval) + " +/- " + str(priorsdev) + " (" + priorname + ")")
    else:
        print("No prior")

    # define number of dimensions (parameters) and specify initial positions of walkers
    # initial values for [H0, Omega_{M0}] are [75, 0.3] plus a random normal disturbance
    ndim = 3
    pos = [75, 0.3, 5] + 1e-3 * np.random.randn(nwalker, ndim)

    # define and run the ensemble sampler
    sampler = emcee.EnsembleSampler(nwalker, ndim, log_probability, args=(z, H, sigmaH))
    sampler.run_mcmc(pos, niter, progress=True)
    print("Final loglikelihood (average of all walkers) = ", str(np.mean(sampler.lnprobability[-1])))

    # produce chain plots (with nuisance parameter M)
    labels = ["H_0", "Omega_{M0}", "M"]
    if plotoutput:
        fig, axes = plt.subplots(3, figsize=(5, 3), sharex='all')
        samples = sampler.get_chain()
        for ii in range(ndim):
            ax = axes[ii]
            ax.plot(samples[:, :, ii], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[ii])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig("M_chain_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # produce corner plots (with nuisance parameter M)
    if plotoutput:
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        fig = corner.corner(flat_samples, labels=labels)
        plt.xlim(0, 0.5)
        plt.savefig("M_corner_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # produce chain plots (without nuisance parameter M)
    labels = ["H_0", "Omega_{M0}"]
    if plotoutput:
        fig, axes = plt.subplots(2, figsize=(5, 3), sharex='all')
        samples = sampler.get_chain(discard=100, thin=15)
        for ii in range(2):  # exclude M
            ax = axes[ii]
            ax.plot(samples[:, :, ii], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[ii])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig("NoM_chain_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # produce corner plots (without nuisance parameter M)
    if plotoutput:
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        flat_samples_NoM = flat_samples[:, 0:2]  # eliminate M
        fig = corner.corner(flat_samples_NoM, labels=labels)
        plt.xlim(0, 0.5)
        plt.savefig("NoM_corner_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # save output to text
    # this includes dataset name, parameter estimates, autocorrelation time
    if saveoutput:
        with open("textoutput.txt", 'a') as f:

            f.write("Dataset/Prior: " + dataname + ", " + priorname + "\n")

            parameters = ["H0", "Omega_M0", "M"]
            for ii in range(ndim):
                mcmc = np.percentile(flat_samples[:, ii], [15, 50, 85])
                q = np.diff(mcmc)  # i.e. (50-15, 85-50)
                f.write("Parameter: " + parameters[ii] + "\n")
                f.write("Value: " + str(mcmc[1]) + "\n")
                f.write("+/-: " + str(q[0]) + ", " + str(q[1]) + "\n\n")

            tau = sampler.get_autocorr_time()
            f.write("Autocorrelation time: " + str(tau) + "\n\n")


# define data sources and priors
datas = ["CC", "CC+SN", "CC+SN+BAO"]
priorvals = [74.22, 69.8, 73.3, 75.35, 67.4, 67.4]
priorsdevs = [1.82, 1.9, 1.7, 1.68, 0.5, 1.1]
priornames = ["Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES"]

# create file for text output
# if already exists (e.g. from previous run, then clear contents of this file)
open('textoutput.txt', 'w').close()

# loop over each dataset and prior
for i in range(len(datas)):

    emcee_LambdaCDM(dataname=datas[i])

    for j in range(len(priornames)):
        emcee_LambdaCDM(prior=True, priorval=priorvals[j], priorsdev=priorsdevs[j], dataname=datas[i], priorname=priornames[j])
