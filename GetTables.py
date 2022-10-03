# import required packages
import os
import numpy as np

# set working directory appropriately by method
os.chdir("~/GaPP_27/Application/1_dgp_new/files/")      # for GPR
os.chdir("~/GaPP_27/Application/3_mcmcdgp_new/files/")  # for MCMC GPR
os.chdir("~/GaPP_27/Application/4_tpr_new/files/")      # for TPR


# returns (x, y, sigma) where x = 0, y = H_0, sigma = std. error
def geth0f(dataname, covname, pname):
    xysigma = open("H_" + dataname + "_" + covname + "_" + pname + ".txt").readline().strip().split()
    xysigma = [float(x) for x in xysigma]
    return xysigma


# sigma distance between two values of H0: see eq. (28) of Briffa et al. (2020)
def calcsigmadistance(h01, h02, sigmanp, sigmap):
    return round((h01 - h02) / np.sqrt(sigmanp ** 2 + sigmap ** 2), 4)


# list of covariance functions and priors
datas = ["CC", "CC+SN", "CC+SN+BAO"]
covfns = ["SquEx", "DoubleSquEx", "Matern32", "Matern52", "Matern72", "Matern92", "Cauchy", "RatQuadratic"]
priors = ["NoPrior", "Riess", "TRGB", "H0LiCOW", "CM", "Planck", "DES"]
priorvals = [None, 74.22, 69.8, 73.3, 75.35, 67.4, 67.4]
priorsdevs = [None, 1.82, 1.9, 1.7, 1.68, 0.5, 1.1]

# get and save H0 and sigma distance values in dictionaries
fxd = {}
fyd = {}
fydr = {}      # rounded values
fsigmad = {}
fsigmadr = {}  # rounded values
sigmadist = {}

# for each covfunction and prior and dataset:
for k in range(len(covfns)):
    for j in range(len(priors)):
        for i in range(len(datas)):

            ctr = tuple([datas[i], priors[j], covfns[k]])

            # get H0 value
            (fxd[ctr], fyd[ctr], fsigmad[ctr]) = geth0f(datas[i], covfns[k], priors[j])
            fydr[ctr] = round(float(fyd[ctr]), 3)
            fsigmadr[ctr] = round(float(fsigmad[ctr]), 4)
            print(covfns[k] + " " + datas[i] + " " + priors[j] + " f: " + str(fydr[ctr]) + " +/- " + str(fsigmadr[ctr]))

            # calculate sigma distance to each prior
            for p in range(len(priors)):
                if p != 0:  # exclude non-prior case
                    ctr2 = tuple([datas[i], priors[j], covfns[k], priors[p]])
                    sigmadist[ctr2] = calcsigmadistance(fyd[ctr], priorvals[p], fsigmad[ctr], priorsdevs[p])
                    print("Sigma Distance to " + priors[p] + ": " + str(sigmadist[ctr2]))

            print


# functions used in table generation
def gety(i1, j1, covfn):
    return "{:.3f}".format(round(fyd[datas[i1], priors[j1], covfn], 3))


def getsigma(i1, j1, covfn):
    return "{:.3f}".format(round(fsigmad[datas[i1], priors[j1], covfn], 3))


def getdist(i1, j1, covfn, p1):
    return "{:.4f}".format(sigmadist[datas[i1], priors[j1], covfn, priors[p1]])


def gettablerow(i1, j1, covfn):
    return str(gety(i1, j1, covfn)) + " \\pm " + \
           str(getsigma(i1, j1, covfn)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 1)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 2)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 3)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 4)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 5)) + "$ & $" + \
           str(getdist(i1, j1, covfn, 6)) + "$ \\\\\n"


def printlatextable(covfn, alert=False):
    if alert:
        print("Printing table for " + covfn)
        print("---------------------------------------------------------")

    print("\\begin{landscape}\\thispagestyle{mylandscape}\n"
          "\\begin{table}[ht]\\centering\\small\n"
          "\\begin{tabular}{cccccccc}\n"
          "\\hline\n"
          "\\textbf{Dataset} &\n"
          "\\textbf{$\\hat{H}_0$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^R)$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^{TRGB})$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^{HW})$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^{CM})$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^P)$} &\n"
          "\\textbf{$d(\\hat{H}_0, \\hat{H}_0^{DES})$} \\\\ \\hline\n"
          "CC                            & $" + gettablerow(0, 0, covfn) +
          "CC+SN                         & $" + gettablerow(1, 0, covfn) +
          "CC+SN+BAO                     & $" + gettablerow(2, 0, covfn) +
          "CC+$\\hat{H}_0^R$             & $" + gettablerow(0, 1, covfn) +
          "CC+SN+$\\hat{H}_0^R$          & $" + gettablerow(1, 1, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^R$      & $" + gettablerow(2, 1, covfn) +
          "CC+$\\hat{H}_0^{TRGB}$        & $" + gettablerow(0, 2, covfn) +
          "CC+SN+$\\hat{H}_0^{TRGB}$     & $" + gettablerow(1, 2, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^{TRGB}$ & $" + gettablerow(2, 2, covfn) +
          "CC+$\\hat{H}_0^{HW}$          & $" + gettablerow(0, 3, covfn) +
          "CC+SN+$\\hat{H}_0^{HW}$       & $" + gettablerow(1, 3, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^{HW}$   & $" + gettablerow(2, 3, covfn) +
          "CC+$\\hat{H}_0^{CM}$          & $" + gettablerow(0, 4, covfn) +
          "CC+SN+$\\hat{H}_0^{CM}$       & $" + gettablerow(1, 4, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^{CM}$   & $" + gettablerow(2, 4, covfn) +
          "CC+$\\hat{H}_0^P$             & $" + gettablerow(0, 5, covfn) +
          "CC+SN+$\\hat{H}_0^P$          & $" + gettablerow(1, 5, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^P$      & $" + gettablerow(2, 5, covfn) +
          "CC+$\\hat{H}_0^{DES}$         & $" + gettablerow(0, 6, covfn) +
          "CC+SN+$\\hat{H}_0^{DES}$      & $" + gettablerow(1, 6, covfn) +
          "CC+SN+BAO+$\\hat{H}_0^{DES}$  & $" + gettablerow(2, 6, covfn) +
          "\\end{tabular}\n"
          "\\caption{MCMC GPR Results with " + str(covfn) + " covariance function}\n"
          "\\label{tab:MCMCdgp" + str(covfn) + "}\n"
          "\\end{table}"
          "\\end{landscape}")

    print


for k in range(len(covfns)):
    printlatextable(covfns[k])
