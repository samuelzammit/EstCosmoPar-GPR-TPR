# coding=utf-8
import matplotlib.pyplot as plt


def plot(z, H, sigmaH, rec, zmin, zmax, plotname, plottitle):
    # plot title
    plt.suptitle(plottitle)

    # reconstructed H(z) values and corresponding prediction errors
    [recz, recH, recsigmaH] = [rec[:, 0], rec[:, 1], rec[:, 2]]

    # plot H(z) on separate plot and save
    plt.figure()
    plt.fill_between(recz, recH + 2 * recsigmaH, recH - 2 * recsigmaH, facecolor='lightgreen')
    plt.fill_between(recz, recH + recsigmaH, recH - recsigmaH, facecolor='lightblue')
    plt.plot(recz, recH)
    plt.errorbar(z, H, sigmaH, color='red', fmt='_')
    plt.title(plottitle)
    plt.xlabel("z")
    plt.ylabel("H(z)")
    plt.savefig("sep_" + plotname + ".pdf")
    plt.close()
