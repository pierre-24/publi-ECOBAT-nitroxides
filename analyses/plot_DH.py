import pandas
import matplotlib.pyplot as plt
import numpy
import argparse

from nitroxides.commons import dG_DH, AU_TO_M, AU_TO_EV


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='DH.pdf')

args = parser.parse_args()

figure = plt.figure(figsize=(10, 5))
ax1, ax2 = figure.subplots(1, 2)

X = numpy.linspace(-2, 1, 31)
for r, c in [(3, 'tab:blue'), (5, 'tab:orange'), (10, 'tab:green')]:
    ax1.plot(10 ** X, dG_DH(1, 0, r / AU_TO_M * 1e-10, r / AU_TO_M * 1e-10, 80, 10 ** X) * AU_TO_EV, '--', color=c)
    ax1.plot(10 ** X, dG_DH(1, 0, r / AU_TO_M * 1e-10, r / AU_TO_M * 1e-10, 35, 10 ** X) * AU_TO_EV, label='a={} Å'.format(r), color=c)

ax1.legend()
ax1.set_xscale('log')
ax1.set_ylabel('$\\Delta G_{DH}^\\star$ (eV)')
ax1.set_xlabel('[X]')

X = numpy.linspace(1, 80, 79 * 2 + 1)
for r, c in [(3, 'tab:blue'), (5, 'tab:orange'), (10, 'tab:green')]:
    ax2.plot(X, dG_DH(1, 0, r / AU_TO_M * 1e-10, r / AU_TO_M * 1e-10, X, 1) * AU_TO_EV, label='a={} Å'.format(r), color=c)

ax2.set_ylabel('$\\Delta G_{DH}^\\star$ (eV)')
ax2.set_xlabel('$\\varepsilon_r$')


plt.tight_layout()
figure.savefig(args.output)
