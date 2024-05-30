import pandas
import matplotlib.pyplot as plt
import numpy
import argparse

from nitroxides.commons import dG_DH, AU_TO_M, AU_TO_EV


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='DH.pdf')

args = parser.parse_args()

figure = plt.figure(figsize=(10, 5))
ax2, ax1 = figure.subplots(1, 2)

X = numpy.linspace(-3, 1, 31)
for r, c in [(3, 'tab:blue'), (5, 'tab:orange'), (10, 'tab:green')]:
    ax1.plot(10 ** X, dG_DH(0, 1, r / AU_TO_M * 1e-10, r / AU_TO_M * 1e-10, 80, 10 ** X) * AU_TO_EV, '--', color=c, label='water' if r == 3 else None)
    ax1.plot(10 ** X, dG_DH(0, 1, r / AU_TO_M * 1e-10, r / AU_TO_M * 1e-10, 35, 10 ** X) * AU_TO_EV, label='acetonitrile' if r == 3 else None)

ax1.set_xscale('log')
ax1.set_ylabel('$\\Delta G_{DH}^\\star$ (eV)')
ax1.set_xlabel('[X] (mol L$^{-1}$)')
ax1.legend()

X = numpy.linspace(1, 80, 79 * 2 + 1)
for r, c in [(3, 'tab:blue'), (5, 'tab:orange'), (10, 'tab:green')]:
    ax2.plot(X, (1 / (2*(r / AU_TO_M * 1e-10)) * (1/X-1)) * AU_TO_EV, label='a={} Å'.format(r), color=c)

ax2.set_ylabel('$\\Delta G_{Born}^\\star$ (eV)')
ax2.set_xlabel('$\\varepsilon_r$')
ax2.legend()


plt.tight_layout()
figure.savefig(args.output)
