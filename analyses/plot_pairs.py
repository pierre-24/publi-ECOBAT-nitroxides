import numpy
import argparse

from nitroxides.commons import AU_TO_EV, AU_TO_M, AU_TO_KJMOL
import matplotlib.pyplot as plt

class System:
    def __init__(self, q: float, a1: float, a2: float, s1: float = 1.0, s2: float = 1.0):
        self.q = q
        self.a1 = a1
        self.a2 = a2
        self.s = s
        
        self.mu = q * (a1 + a2) * s1
        self.a = (a1**3 + a2**3)**(1/3) * s2
    
    @staticmethod
    def _e_born_solvation(q: float, a: float, epsr: float, epsi: float = 1.0) -> float:
        """Solvatation energy from Born theory"""
        return q ** 2 / (2*a) * (1/epsr - 1/epsi)
    
    @staticmethod
    def _e_coulomb(q1: float, q2: float, r: float, eps: float) -> float:
        """Coulombic interaction between two charges in a given medium"""
        return q1 * q2 / (r*eps)
    
    @staticmethod
    def _e_coulomb2(q1: float, q2: float, mu: float, eps: float) -> float:
        """Coulombic interaction / formation of a dipole"""
        return q1 * q2 * numpy.abs(q1) / (eps * mu)
    
    @staticmethod
    def _e_dipole_onsager(mu: float, a: float, epsr: float, epsi: float = 1.0) -> float:
        """Solvatation energy of a dipole in a cavity, according to Onsager"""
        return -3*(epsr-epsi)/((2*epsr+1)*(2*epsi+1))*mu**2/a**3
    
    def e_born_solvation(self, epsr: float, epsi: float = 1.0) -> float:
        return System._e_born_solvation(self.q, self.a1, epsr, epsi) + System._e_born_solvation(-self.q, self.a2, epsr, epsi)
    
    def e_coulomb(self, epsi: float = 1.0) -> float:
        return System._e_coulomb(self.q, -self.q, self.a1 + self.a2, epsi)
    
    def e_coulomb2(self, eps: float) -> float:
        return System._e_coulomb2(self.q, -self.q, self.mu, eps)
    
    def e_dipole_onsager(self, epsr, epsi: float = 1.0) -> float:
        return System._e_dipole_onsager(self.mu, self.a, epsr, epsi)
    
    def e_pair(self, epsr: float, epsi: float = 1.0) -> float:
        return self.e_coulomb2(epsi) + self.e_dipole_onsager(epsr, epsi) - self.e_born_solvation(epsr, epsi)
        
a = 3.0 / AU_TO_M * 1e-10
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

def logK(dG: float):
    return  numpy.log10(numpy.exp(-dG * AU_TO_KJMOL / (R * T)))

for s, color in [(1.0, 'tab:blue'), (1.2, 'tab:orange'), (1.4, 'tab:green')]:
    ax1.plot(xi, logK(System(1, a1, a2, s1 = 1.0, s2 = s).e_pair(35)), label='$s_2$={}'.format(s), color=color)
    ax1.plot(xi, logK(System(1, a1, a2, s1= 1.0, s2 = s).e_pair(80)), '--', color=color)
    ax2.plot(xi, logK(System(1, a1, a2, s1 = 0.7, s2 = s).e_pair(35)), label='$s_2$={}'.format(s), color=color)
    ax2.plot(xi, logK(System(1, a1, a2, s1= 0.7, s2 = s).e_pair(80)), '--', color=color)

[ax.set_xlabel('$\\chi = a_1$ / $a_2$') for ax in (ax1, ax2)]
[ax.set_xlim(MI, MX) for ax in (ax1, ax2)]
ax1.set_ylabel('log$_{10}$($K_{pair}$)')

ax1.legend()

ax1.text(2, 10, '$s_1$=1', fontsize=14)
ax2.text(2, 10, '$s_1$=0.7', fontsize=14)

plt.tight_layout()
fig.savefig(args.output)
