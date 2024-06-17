import numpy
import argparse

from nitroxides.commons import AU_TO_EV, AU_TO_ANG, AU_TO_KJMOL
from nitroxides.pairs import IonPair as System
import matplotlib.pyplot as plt
        
a = 3.0 / AU_TO_ANG
MI = 0.75
MX = 3
N = 81
eps = 35

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='pair.pdf')

args = parser.parse_args()

fig = plt.figure(figsize=(8, 5))
ax1, ax2 = fig.subplots(1, 2, sharey=True)

xi = numpy.linspace(MI, MX, N)
a1 = numpy.repeat(a, N)
a2 = xi * a1

[ax.plot([MI,MX], [0, 0], color='grey') for ax in (ax1, ax2)]

R = 8.3145e-3
T=298.15

def pK(dG: float):
    return  -numpy.log10(numpy.exp(-dG * AU_TO_KJMOL / (R * T)))

for s, color in [(1.0, 'tab:blue'), (1.2, 'tab:orange'), (1.4, 'tab:green')]:
    ax1.plot(xi, pK(System(1, a1, a2, s1 = 1.0, s2 = s).e_pair(35)), label='$s_2$={}'.format(s), color=color)
    ax1.plot(xi, pK(System(1, a1, a2, s1= 1.0, s2 = s).e_pair(80)), '--', color=color)
    ax2.plot(xi, pK(System(1, a1, a2, s1 = 0.7, s2 = s).e_pair(35)), label='$s_2$={}'.format(s), color=color)
    ax2.plot(xi, pK(System(1, a1, a2, s1= 0.7, s2 = s).e_pair(80)), '--', color=color)

[ax.set_xlabel('$\\chi = a_1$ / $a_2$') for ax in (ax1, ax2)]
[ax.set_xlim(MI, MX) for ax in (ax1, ax2)]
ax1.set_ylabel('pK$_{pair}$')

ax1.legend()

ax1.text(2, -10, '$s_1$=1', fontsize=14)
ax2.text(2, -10, '$s_1$=0.7', fontsize=14)

plt.tight_layout()
fig.savefig(args.output)
