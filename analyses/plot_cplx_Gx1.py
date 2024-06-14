import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import AU_TO_ANG, AU_TO_KJMOL, H_NME4, H_BF4, S_NME4, S_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE, dG_DH_cplx_Kx1, EPSILON_R

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹

def prepare_data(data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    
    dH_DH_k01 = dG_DH_cplx_Kx1(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_A_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dH_DH_k11 = dG_DH_cplx_Kx1(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dH_DH_k21 = dG_DH_cplx_Kx1(subdata['z'] - 1, subdata['z'] - 1, 1, subdata['r_A_red'] / AU_TO_ANG, subdata['r_AX_red'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    subdata.insert(1, 'dH_cplx_ox', (subdata['H_cplx_ox'] - H_BF4[solvent] + dH_DH_k01) * AU_TO_KJMOL)
    subdata.insert(1, 'dH_cplx_rad', (subdata['H_cplx_rad'] - H_NME4[solvent] + dH_DH_k11) * AU_TO_KJMOL)
    subdata.insert(1, 'dH_cplx_red', (subdata['H_cplx_red'] - H_NME4[solvent] + dH_DH_k21) * AU_TO_KJMOL)
    
    subdata.insert(1, 'dS_cplx_ox', (subdata['S_cplx_ox'] - S_BF4[solvent]) * AU_TO_KJMOL)
    subdata.insert(1, 'dS_cplx_rad', (subdata['S_cplx_rad'] - S_NME4[solvent]) * AU_TO_KJMOL)
    subdata.insert(1, 'dS_cplx_red', (subdata['S_cplx_red'] - S_NME4[solvent]) * AU_TO_KJMOL)
    
    return subdata

def plot_Gx1(ax, data: pandas.DataFrame, state: str, family: str, color: str):
    subdata = data[data['family'] == family]
    
    x = [int(x.replace('mol_', '')) for x in subdata['name']]
    ax.bar(x, subdata['dH_cplx_{}'.format(state)], color=color, label=family.replace('Family.', ''))
    ax.bar(x, -T * subdata['dS_cplx_{}'.format(state)], edgecolor=color, color='none')
    ax.plot(x, subdata['dH_cplx_{}'.format(state)] -T * subdata['dS_cplx_{}'.format(state)], 'o', color=color)
    

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

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 10))
ax1, ax2, ax3 = figure.subplots(3, 1, sharey=True, sharex=True)

subdata_wa = prepare_data(data, 'water')

plot_Gx1(ax1, subdata_wa, 'ox', 'Family.AMO', 'tab:pink')
plot_Gx1(ax1, subdata_wa, 'ox', 'Family.P6O', 'tab:blue')
plot_Gx1(ax1, subdata_wa, 'ox', 'Family.P5O', 'black')
plot_Gx1(ax1, subdata_wa, 'ox', 'Family.IIO', 'tab:green')
plot_Gx1(ax1, subdata_wa, 'ox', 'Family.APO', 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(5, -40, "N$^+$A$^-$", fontsize=18)

plot_Gx1(ax2, subdata_wa, 'rad', 'Family.AMO', 'tab:pink')
plot_Gx1(ax2, subdata_wa, 'rad', 'Family.P6O', 'tab:blue')
plot_Gx1(ax2, subdata_wa, 'rad', 'Family.P5O', 'black')
plot_Gx1(ax2, subdata_wa, 'rad', 'Family.IIO', 'tab:green')
plot_Gx1(ax2, subdata_wa, 'rad', 'Family.APO', 'tab:red')
ax2.text(5, -40, "N$^\\bullet$C$^+$", fontsize=18)

plot_Gx1(ax3, subdata_wa, 'red', 'Family.AMO', 'tab:pink')
plot_Gx1(ax3, subdata_wa, 'red', 'Family.P6O', 'tab:blue')
plot_Gx1(ax3, subdata_wa, 'red', 'Family.P5O', 'black')
plot_Gx1(ax3, subdata_wa, 'red', 'Family.IIO', 'tab:green')
plot_Gx1(ax3, subdata_wa, 'red', 'Family.APO', 'tab:red')
ax3.text(5, -40, "N$^-$C$^+$", fontsize=18)

[ax.set_ylabel('Thermochemical contribution (kJ mol$^{-1}$)') for ax in [ax1, ax2, ax3]]
[ax.xaxis.set_minor_locator(MultipleLocator(2)) for ax in [ax1, ax2, ax3]]
[ax.grid(which='both', axis='x') for ax in [ax1, ax2, ax3]]
[ax.plot([0, 62], [0, 0], '-', color='grey') for ax in [ax1, ax2, ax3]]

ax3.set_xlabel('Molecule id') 

plt.tight_layout()
figure.savefig(args.output)
