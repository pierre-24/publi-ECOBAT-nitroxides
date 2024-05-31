import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import argparse

from nitroxides.commons import AU_TO_ANG, AU_TO_EV, G_NME4, G_BF4, RADII_BF4, RADII_NME4, dG_DH_cplx_Kx1, AU_TO_KJMOL
from nitroxides.pairs import IonPair

S1=1
S2 = 1.5

def plot_r_Kx1(ax, data: pandas.DataFrame, family: str, solvent: str, epsilon_r: float, color: str, has_cation: bool = False, has_anion: bool = False):
    subdata = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_anion'] == has_anion) & (data['has_cation'] == has_cation)] 
    
    if has_anion:
        dG_DH = dG_DH_cplx_Kx1(subdata['z'] + 1, subdata['z'], -1, subdata['r_A'] / AU_TO_ANG, subdata['r_AX'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, epsilon_r)  # K_01
        dG = (subdata['G_cplx'] - G_BF4[solvent] + dG_DH) * AU_TO_EV
        dGp = IonPair(
            q=1, 
            a1=subdata['r_A'] / AU_TO_ANG, 
            a2=RADII_BF4[solvent] / AU_TO_ANG, 
            s1=subdata['d_OX'] / (subdata['r_A'] + RADII_BF4[solvent]), 
            s2=subdata['r_AX']/(subdata['r_A'] **3 + RADII_BF4[solvent]**3)**(1/3)
        ).e_pair(epsilon_r)
    else:
        dG_DH = dG_DH_cplx_Kx1(subdata['z'] - 1, subdata['z'], 1, subdata['r_A'] / AU_TO_ANG, subdata['r_AX'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, epsilon_r)  # K_21
        dG = (subdata['G_cplx'] - G_NME4[solvent] + dG_DH) * AU_TO_EV
        dGp = IonPair(
            q=-1, 
            a1=subdata['r_A'] / AU_TO_ANG, 
            a2=RADII_NME4[solvent] / AU_TO_ANG, 
            s1=subdata['d_OX'] / (subdata['r_A'] + RADII_NME4[solvent]), 
            s2=subdata['r_AX']/(subdata['r_A'] **3 + RADII_NME4[solvent]**3)**(1/3)
        ).e_pair(epsilon_r)
    
    ax.plot(dGp, dG, 'o', color=color, label=family.replace('Family.', ''))
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_cplx_Kx1.csv')
parser.add_argument('-o', '--output', default='Data_cplx_r_Kx1.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(8, 6))
ax1, ax2 = figure.subplots(2, 1, sharey=True)

plot_r_Kx1(ax1, data, 'Family.AMO', 'acetonitrile', 35.,'tab:pink', has_anion=True)
plot_r_Kx1(ax1, data, 'Family.P6O', 'acetonitrile', 35.,'tab:blue', has_anion=True)
plot_r_Kx1(ax1, data, 'Family.P5O', 'acetonitrile', 35., 'black', has_anion=True)
plot_r_Kx1(ax1, data, 'Family.IIO', 'acetonitrile', 35., 'tab:green', has_anion=True)
plot_r_Kx1(ax1, data, 'Family.APO', 'acetonitrile', 35., 'tab:red', has_anion=True)

ax1.legend(ncols=5)
# ax1.text(38, 1, "Water", fontsize=18)

plot_r_Kx1(ax2, data, 'Family.AMO', 'acetonitrile', 35.,'tab:pink', has_cation=True)
plot_r_Kx1(ax2, data, 'Family.P6O', 'acetonitrile', 35.,'tab:blue', has_cation=True)
plot_r_Kx1(ax2, data, 'Family.P5O', 'acetonitrile', 35., 'black', has_cation=True)
plot_r_Kx1(ax2, data, 'Family.IIO', 'acetonitrile', 35., 'tab:green', has_cation=True)
plot_r_Kx1(ax2, data, 'Family.APO', 'acetonitrile', 35., 'tab:red', has_cation=True)

#ax2.text(38, -2, "Acetonitrile", fontsize=18)

[ax.set_ylabel('$\\Delta G^\\star_{pair}$ (eV)') for ax in [ax1, ax2]]

# ax2.set_xlabel('$d$ (Å)')

plt.tight_layout()
figure.savefig(args.output)
