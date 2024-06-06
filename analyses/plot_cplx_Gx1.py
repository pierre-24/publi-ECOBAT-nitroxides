import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import AU_TO_ANG, AU_TO_KJMOL, G_NME4, G_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE, dG_DH_cplx_Kx1, EPSILON_R

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹

def prepare_data(data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    
    dG_DH_k01 = dG_DH_cplx_Kx1(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_A_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k11 = dG_DH_cplx_Kx1(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k21 = dG_DH_cplx_Kx1(subdata['z'] - 1, subdata['z'] - 1, 1, subdata['r_A_red'] / AU_TO_ANG, subdata['r_AX_red'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    subdata.insert(1, 'dG_cplx_ox', (subdata['G_cplx_ox']- G_BF4[solvent] + dG_DH_k01) * AU_TO_KJMOL)
    subdata.insert(1, 'dG_cplx_rad', (subdata['G_cplx_rad'] - G_NME4[solvent] + dG_DH_k11) * AU_TO_KJMOL)
    subdata.insert(1, 'dG_cplx_red', (subdata['G_cplx_red'] - G_NME4[solvent] + dG_DH_k21) * AU_TO_KJMOL)
    
    return subdata

def plot_Gx1(ax, data: pandas.DataFrame, family: str, color: str):
    subdata = data[data['family'] == family]
    
    x = [int(x.replace('mol_', '')) for x in subdata['name']]
    
    ax.plot(x, subdata['dG_cplx_ox'], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(x, subdata['dG_cplx_rad'], '^', color=color)
    ax.plot(x, subdata['dG_cplx_red'], 's', color=color)

def plot_helpline(ax, data):
    x = [int(x.replace('mol_', '')) for x in data['name']]
    ax.plot(x, data['dG_cplx_ox'], '--', color='black', linewidth=0.8)
    

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

figure = plt.figure(figsize=(10, 8))
ax1, ax2 = figure.subplots(2, 1, sharey=True, sharex=True)

subdata_wa = prepare_data(data, 'water')
plot_helpline(ax1, subdata_wa)

plot_Gx1(ax1, subdata_wa, 'Family.AMO', 'tab:pink')
plot_Gx1(ax1, subdata_wa, 'Family.P6O', 'tab:blue')
plot_Gx1(ax1, subdata_wa, 'Family.P5O', 'black')
plot_Gx1(ax1, subdata_wa, 'Family.IIO', 'tab:green')
plot_Gx1(ax1, subdata_wa, 'Family.APO', 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(38, -10, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')
ax1.plot([0, 62], [0, 0], '-', color='grey')

subdata_ac = prepare_data(data, 'acetonitrile')
plot_helpline(ax2, subdata_ac)

plot_Gx1(ax2, subdata_ac, 'Family.P6O', 'tab:blue')
plot_Gx1(ax2, subdata_ac, 'Family.P5O', 'black')
plot_Gx1(ax2, subdata_ac, 'Family.IIO', 'tab:green')
plot_Gx1(ax2, subdata_ac, 'Family.APO', 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
ax2.text(38, -10, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('$\\Delta G^\\star_{cplx}$ (kJ mol$^{-1}$)') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)
