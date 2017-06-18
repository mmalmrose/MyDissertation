import matplotlib
import matplotlib.pyplot as plt
def plot_chisquare_better(mjdarr, chiarr, title, name, plotout):
    fig = plt.figure(17, figsize=(10,7))
    plt.plot(mjdarr, chiarr, color = (1,0,0), marker='o', ls='')
    plt.xlabel('MJD', fontsize=20)
    plt.ylabel(r'$\chi_{reduced}^{2}$', fontsize=20)
    plt.title(title)
    plt.savefig(plotout+name)
    plt.close(fig)
    plt.cla()
