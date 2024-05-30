import numpy
import argparse

from nitroxides.commons import AU_TO_KJMOL
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='CIP.pdf')

args = parser.parse_args()

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()

MI = -2
MX = 1
N = 62

X = 10**numpy.linspace(MI, MX, N)

ax.plot([10**MI,10**MX], [0, 0], color='grey')

R = 8.3145
T= 298.15
F =  9.64853321233100184e4

def Ef_ox(X: float, k01: float, k02: float, k12: float, E0: float = 0):
    return E0 + R * T / F * numpy.log((1+k12*X**2) / (1+k01*X+k02*X**2))
    
def Ef_red(X: float, k12: float, k21: float, k22: float, E0: float = 0):
    return E0 + R * T / F * numpy.log((1+k21*X+k22*X**2) / (1+k12*X**2))

for kx1, kx2, color in [(1, 1e-3, 'tab:blue'), (1e-3, 1, 'tab:orange'), (1, 1e-1, 'tab:green'), (1e-1, 1, 'tab:red')]:
    ax.plot(X, Ef_ox(X, kx1, kx2, kx2), color=color, label='$K_{{x1}}$={}, $K_{{x2}}={}$'.format(kx1, kx2))
    ax.plot(X, Ef_red(X, kx2, kx1, kx2), '--', color=color)

ax.set_xlabel('[X] (mol L$^{-1}$)')
ax.set_ylabel('$E^f_{abs}$ (V)')
ax.set_xscale('log')
ax.set_xlim(10**MI, 10**MX)
ax.legend()

plt.tight_layout()
fig.savefig(args.output)
