# import required packages
import os
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

os.chdir("~/GaPP_27/Application/2_mcmc_emcee_LambdaCDM/files/")


# emcee_LambdaCDM carries out MCMC inference using the LCDM model with parameters H_0 and Omega_{M0}
def emcee_LambdaCDM(prior=False, priorval=None, priorsdev=None, nwalker=200, niter=100000, plotoutput=True,
                    saveoutput=True, dataname, priorname):
    if __name__ == "__main__":

        # load Hubble data
        # including covariance matrix for SN readings
        (zCC, HCC, sigmaHCC) = np.loadtxt("~/GaPP_27/Application/data/Hdata_CC.txt", unpack=True)

        (zSN, ESN, sigmaESN) = np.loadtxt("~/GaPP_27/Application/data/Hdata_SN_E.txt", unpack=True)
        covSN = np.loadtxt("~/GaPP_27/Application/data/covmatrix_SN_E.txt", unpack=True)

        (zBAO, HBAO, sigmaHBAO) = np.loadtxt("~/GaPP_27/Application/data/Hdata_BAO.txt", unpack=True)

        # prepend priors
        # adding this to CC dataset as it is used in all cases
        # (e.g. we do not have a run of just SN+BAO)
        if prior:
            zCC = np.insert(zCC, 0, 0, axis=0)
            HCC = np.insert(HCC, 0, priorval, axis=0)
            sigmaHCC = np.insert(sigmaHCC, 0, priorsdev, axis=0)
        else:
            priorval = None
            priorsdev = None

        # obtain (z, H, sigmaH) by concatenating datasets based on supplied dataname
        (z, H, sigmaH) = (zCC, HCC, sigmaHCC)
        if dataname == "CC+SN" or dataname == "CC+SN+BAO":
            (z, H, sigmaH) = (np.append(z, zSN), np.append(H, ESN), np.append(sigmaH, sigmaESN))
            if dataname == "CC+SN+BAO":
                (z, H, sigmaH) = (np.append(z, zBAO), np.append(H, HBAO), np.append(sigmaH, sigmaHBAO))

        # function definition
        # prior (mostly uninformative)
        def log_prior(theta):
            H0, Omega_M0 = theta
            if 50 < H0 < 100 and 0.1 < Omega_M0 < 0.5:
                return 0.0
            return -np.inf

        # likelihood (CC, SN, BAO components)
        def log_likelihood_CC(theta):
            H0, Omega_M0 = theta
            Hpred = H0 * np.sqrt(Omega_M0 * (1 + zCC) ** 3 + (1 - Omega_M0))
            return -0.5 * np.sum((HCC - Hpred) ** 2 / sigmaHCC ** 2)

        def log_likelihood_SN(theta):
            H0, Omega_M0 = theta
            Epred = np.sqrt(Omega_M0 * (1 + zSN) ** 3 + (1 - Omega_M0))
            CSNinv = np.linalg.inv(covSN)
            return -0.5 * np.sum((ESN - Epred) * CSNinv * (ESN - Epred))

        def log_likelihood_BAO(theta):
            H0, Omega_M0 = theta
            Hpred = H0 * np.sqrt(Omega_M0 * (1 + zBAO) ** 3 + (1 - Omega_M0))
            return -0.5 * np.sum((HBAO - Hpred) ** 2 / sigmaHBAO ** 2)

        # posterior
        def log_probability(theta, z, H, sigmaH):
            logpri = log_prior(theta)
            if not np.isfinite(logpri):
                return -np.inf
            logpost = {
                "CC": logpri + log_likelihood_CC(theta),
                "CC+SN": logpri + log_likelihood_CC(theta) + log_likelihood_SN(theta),
                "CC+SN+BAO": logpri + log_likelihood_CC(theta) + log_likelihood_SN(theta) + log_likelihood_BAO(theta)
            }.get(dataname)
            return logpost

    print("\nDataset: " + dataname)

    # print information on dataset, covariance function, prior
    if priorval is not None and priorsdev is not None:
        print("Prior: " + str(priorval) + " +/- " + str(priorsdev) + " (" + priorname + ")")
    else:
        print("No prior")

    # define number of dimensions (parameters) and specify initial positions of walkers
    # initial values for [H_0, Omega_{M0}] are [75, 0.3] plus a random normal disturbance
    ndim = 2  # corresponding to [H_0, Omega_{M0}]
    pos = [75, 0.3] + 1e-3 * np.random.randn(nwalker, ndim)

    # define and run the ensemble sampler
    sampler = emcee.EnsembleSampler(nwalker, ndim, log_probability, args=(z, H, sigmaH))
    sampler.run_mcmc(pos, niter, progress=True)
    print("Final loglikelihood (average of all walkers) = ", str(np.mean(sampler.lnprobability[-1])))

    # produce chain plots
    labels = ["H_0", "Omega_{M0}"]
    if plotoutput:
        fig, axes = plt.subplots(2, figsize=(5, 3), sharex='all')
        samples = sampler.get_chain()
        for ii in range(ndim):
            ax = axes[ii]
            ax.plot(samples[:, :, ii], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[ii])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.savefig("chain_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # produce corner plots
    if plotoutput:
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        fig = corner.corner(flat_samples, labels=labels)
        plt.xlim(0, 0.5)
        plt.savefig("corner_" + dataname + "_" + priorname + ".pdf")
        plt.close()

    # save output to text
    # this includes dataset name, parameter estimates, autocorrelation time
    if saveoutput:
        with open("textoutput.txt", 'a') as f:

            f.write("Dataset/Prior: " + dataname + ", " + priorname + "\n")

            parameters = ["H_0", "Omega_M0"]
            for ii in range(ndim):
                mcmc = np.percentile(flat_samples[:, ii], [15, 50, 85])
                q = np.diff(mcmc)
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
# if already exists (e.g. from previous run), then clear contents of this file
open('textoutput.txt', 'w').close()

# loop over each dataset and prior
for i in range(len(datas)):

    emcee_LambdaCDM(dataname=datas[i], priorname="NoPrior")

    for j in range(len(priornames)):
        emcee_LambdaCDM(prior=True, priorval=priorvals[j], priorsdev=priorsdevs[j], dataname=datas[i], priorname=priornames[j])
