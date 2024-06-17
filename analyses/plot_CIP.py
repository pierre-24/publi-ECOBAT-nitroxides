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

def Ef_ox(X: float, k01: float, k02: float, k11: float, k12: float, E0: float = 0):
    return E0 + R * T / F * numpy.log((1+k11*X+k12*X**2) / (1+k01*X+k02*X**2))
    
def Ef_red(X: float, k11: float, k12: float, k21: float, k22: float, E0: float = 0):
    return E0 + R * T / F * numpy.log((1+k21*X+k22*X**2) / (1+k11*X+k12*X**2))

for ki1, kip1, ki2, kip2, color in [(1, 1e-2, 1,1, 'tab:green'), (1e-2, 1, 1e-2, 1, 'tab:blue'), (1, 1e-2, 1, 1e-2, 'tab:red'), (1, 1, 1, 1e-2, 'tab:orange')]:
    ax.plot(X, Ef_ox(X, ki1, ki2, kip1, kip2), color=color, label='$K_{{01}}$={}, $K_{{11}}$={}, $K_{{02}}$={}, $K_{{12}}$={}'.format(ki1, kip1, ki2, kip2))

ax.set_xlabel('[X] (mol L$^{-1}$)')
ax.set_ylabel('$E^f_{abs}(N$+$|N$^\\bullet$)$ (V)')
ax.set_xscale('log')
ax.set_xlim(10**MI, 10**MX)
ax.legend()

plt.tight_layout()
fig.savefig(args.output)
