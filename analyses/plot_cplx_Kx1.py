import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import G_DH, AU_TO_M, AU_TO_KJMOL, kappa2, G_NME4, G_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹

def dG_DH_cplx(z_reac: int, z_prod: int, z_ct: int, a_reac: float, a_prod: float, a_ct: float, epsilon_r: float, c_act: float = C_NITROXIDE, c_elt: float = 1, z_elt: float = 1):
    # complexation reaction correction!
    kappa_reac = numpy.sqrt(kappa2(c_act, z_reac, epsilon_r) + kappa2(c_act, -z_reac, epsilon_r) + kappa2(c_elt, z_elt, epsilon_r) + kappa2(c_elt, -z_elt, epsilon_r))  # in bohr⁻¹
    kappa_prod = numpy.sqrt(kappa2(c_act, z_prod, epsilon_r) + kappa2(c_act, -z_prod, epsilon_r) + kappa2(c_elt, z_elt, epsilon_r) + kappa2(c_elt, -z_elt, epsilon_r))  # in bohr⁻¹
    
    dG = G_DH(z_prod, kappa_prod, epsilon_r, a_prod) - G_DH(z_reac, kappa_reac, epsilon_r, a_reac) -  G_DH(z_ct, kappa_reac, epsilon_r, a_ct) # in au
    
    return dG

def plot_Kx1(ax, data: pandas.DataFrame, family: str, solvent: str, epsilon_r: float, color: str):
    subdata_k01 = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_anion'] == True)] # K_01 → N+ + A-
    subdata_k21 = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_cation'] == True)] # K_21 → N- + C+
    
    dG_DH_k01 = dG_DH_cplx(subdata_k01['z'] + 1, subdata_k01['z'], -1, subdata_k01['r_A'], subdata_k01['r_AX'], RADII_BF4[solvent], epsilon_r)
    dG_DH_k21 = dG_DH_cplx(subdata_k21['z'] - 1, subdata_k21['z'], 1, subdata_k21['r_A'], subdata_k21['r_AX'], RADII_NME4[solvent], epsilon_r)
    
    dG_k01 = (subdata_k01['G_cplx'] - G_BF4[solvent]+ dG_DH_k01) * AU_TO_KJMOL
    dG_k21 = (subdata_k21['G_cplx'] - G_NME4[solvent] + dG_DH_k21) * AU_TO_KJMOL
    
    k01 = numpy.exp(-dG_k01 / (R * T))
    k21 = numpy.exp(-dG_k21 / (R * T))
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k01['name']], numpy.log10(k01), 'o', color=color, label=family.replace('Family.', ''))
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k21['name']], numpy.log10(k21), 's', color=color)

def helpline_K01(ax, data: pandas.DataFrame, solvent: str, epsilon_r: float, color: str = 'black'):
    subdata_k01 = data[(data['solvent'] == solvent) & (data['has_anion'] == True)] # K_01 → N+ + A-
    
    dG_DH_k01 = dG_DH_cplx(subdata_k01['z'] + 1, subdata_k01['z'], -1, subdata_k01['r_A'], subdata_k01['r_AX'], RADII_BF4[solvent], epsilon_r)
    
    dG_k01 = (subdata_k01['G_cplx'] - G_BF4[solvent]+ dG_DH_k01) * AU_TO_KJMOL
    
    k01 = numpy.exp(-dG_k01 / (R * T))
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k01['name']], numpy.log10(k01), '--', color=color, linewidth=0.75)
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_cplx_Kx1.csv')
parser.add_argument('-o', '--output', default='Data_cplx_Kx1.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 8))
ax1, ax2 = figure.subplots(2, 1)

helpline_K01(ax1, data, 'water', 80, 'black')

plot_Kx1(ax1, data, 'Family.AMO', 'water', 80.,'tab:pink')
plot_Kx1(ax1, data, 'Family.P6O', 'water', 80.,'tab:blue')
plot_Kx1(ax1, data, 'Family.P5O', 'water', 80., 'black')
plot_Kx1(ax1, data, 'Family.IIO', 'water', 80., 'tab:green')
plot_Kx1(ax1, data, 'Family.APO', 'water', 80., 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(38, 1, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')
ax1.plot([0, 62], [0, 0], '-', color='grey')

helpline_K01(ax2, data, 'acetonitrile', 35, 'black')

plot_Kx1(ax2, data, 'Family.P6O', 'acetonitrile', 35.,'tab:blue')
plot_Kx1(ax2, data, 'Family.P5O', 'acetonitrile', 35., 'black')
plot_Kx1(ax2, data, 'Family.IIO', 'acetonitrile', 35., 'tab:green')
plot_Kx1(ax2, data, 'Family.APO', 'acetonitrile', 35., 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
ax2.text(38, -2, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('log$_{10}$(K)') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)
