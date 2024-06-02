import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import AU_TO_ANG, AU_TO_KJMOL, G_NME4, G_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE, dG_DH_cplx_Kx1, EPSILON_R
from nitroxides.tex import format_longtable

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹

def plot_Kx1(ax, data: pandas.DataFrame, family: str, solvent: str, color: str):
    subdata_k01 = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_anion'] == True)] # K_01 → N+ + A-
    subdata_k21 = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_cation'] == True)] # K_21 → N- + C+
    
    dG_DH_k01_0 = dG_DH_cplx_Kx1(subdata_k01['z'] + 1, subdata_k01['z'], -1, subdata_k01['r_A'] / AU_TO_ANG, subdata_k01['r_AX'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent], c_elt=0.1)
    dG_DH_k21_0 = dG_DH_cplx_Kx1(subdata_k21['z'] - 1, subdata_k21['z'], 1, subdata_k21['r_A'] / AU_TO_ANG, subdata_k21['r_AX'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent], c_elt=0.1)
    
    dG_DH_k01 = dG_DH_cplx_Kx1(subdata_k01['z'] + 1, subdata_k01['z'], -1, subdata_k01['r_A'] / AU_TO_ANG, subdata_k01['r_AX'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k21 = dG_DH_cplx_Kx1(subdata_k21['z'] - 1, subdata_k21['z'], 1, subdata_k21['r_A'] / AU_TO_ANG, subdata_k21['r_AX'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    dG_k01_0 = (subdata_k01['G_cplx'] - G_BF4[solvent] + dG_DH_k01_0) * AU_TO_KJMOL
    dG_k21_0 = (subdata_k21['G_cplx'] - G_NME4[solvent] + dG_DH_k21_0) * AU_TO_KJMOL
    
    dG_k01 = (subdata_k01['G_cplx'] - G_BF4[solvent] + dG_DH_k01) * AU_TO_KJMOL
    dG_k21 = (subdata_k21['G_cplx'] - G_NME4[solvent] + dG_DH_k21) * AU_TO_KJMOL
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k01['name']], dG_k01, 'o', color=color, label=family.replace('Family.', ''))
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k21['name']], dG_k21, 's', color=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k01['name']], dG_k01_0, 'o', markerfacecolor='none', markeredgecolor=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata_k21['name']], dG_k21_0, 's', markerfacecolor='none', markeredgecolor=color)

def make_table(f, data: pandas.DataFrame, solvent: str):
    subdata_k01 = data[(data['solvent'] == solvent) & (data['has_anion'] == True)] # K_01 → N+ + A-
    subdata_k21 = data[(data['solvent'] == solvent) & (data['has_cation'] == True)] # K_21 → N- + C+
    
    dG_DH_k01 = dG_DH_cplx_Kx1(subdata_k01['z'] + 1, subdata_k01['z'], -1, subdata_k01['r_A'] / AU_TO_ANG, subdata_k01['r_AX'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k21 = dG_DH_cplx_Kx1(subdata_k21['z'] - 1, subdata_k21['z'], 1, subdata_k21['r_A'] / AU_TO_ANG, subdata_k21['r_AX'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    subdata_k01.insert(1, 'dG_cplx', (subdata_k01['G_cplx'] - G_BF4[solvent] + dG_DH_k01) * AU_TO_KJMOL)
    subdata_k21.insert(1, 'dG_cplx', (subdata_k21['G_cplx'] - G_NME4[solvent] + dG_DH_k21) * AU_TO_KJMOL)
    
    subdata = subdata_k01.join(subdata_k21.set_index('name'), on='name', lsuffix='k01', rsuffix='k21', how='inner')
    
    f.write(format_longtable(
        subdata, 
        titles=['', '$a_{\\ce{NA}}$', '$\\Delta{G}_{cplx}^\\star$', '', '$a_{\\ce{NC}}$','$\\Delta{G}_{cplx}^\\star$'], 
        line_maker=lambda r: [
            r['name'].replace('mol_', ''), 
            '{:.2f}'.format(r['r_AXk01']), 
            '{:.1f}'.format(r['dG_cplxk01']), 
            '',
            '{:.2f}'.format(r['r_AXk21']), 
            '{:.1f}'.format(r['dG_cplxk21']), 
        ]
    ))
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_cplx_Kx1.csv')
parser.add_argument('-o', '--output', default='Data_cplx_Gx1.pdf')
parser.add_argument('-t', '--table')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 8))
ax1, ax2 = figure.subplots(2, 1, sharey=True, sharex=True)

plot_Kx1(ax1, data, 'Family.AMO', 'water', 'tab:pink')
plot_Kx1(ax1, data, 'Family.P6O', 'water', 'tab:blue')
plot_Kx1(ax1, data, 'Family.P5O', 'water', 'black')
plot_Kx1(ax1, data, 'Family.IIO', 'water', 'tab:green')
plot_Kx1(ax1, data, 'Family.APO', 'water', 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(38, 0.5, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')
ax1.plot([0, 62], [0, 0], '-', color='grey')

plot_Kx1(ax2, data, 'Family.P6O', 'acetonitrile', 'tab:blue')
plot_Kx1(ax2, data, 'Family.P5O', 'acetonitrile', 'black')
plot_Kx1(ax2, data, 'Family.IIO', 'acetonitrile', 'tab:green')
plot_Kx1(ax2, data, 'Family.APO', 'acetonitrile', 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
ax2.text(38, 0.5, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('$\\Delta G^\\star_{cplx}$ (kJ mol$^{-1}$)') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, data, 'water')
        make_table(f, data, 'acetonitrile')