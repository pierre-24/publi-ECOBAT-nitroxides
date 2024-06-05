import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import AU_TO_ANG, AU_TO_KJMOL, G_NME4, G_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE, dG_DH_cplx_Kx2, EPSILON_R
from nitroxides.tex import format_longtable

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹

def plot_Kx2(ax, data: pandas.DataFrame, family: str, solvent: str, color: str):
    subdata = data[(data['family'] == family) & (data['solvent'] == solvent)]
    
    dG_DH_k02 = dG_DH_cplx_Kx2(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_A_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k12 = dG_DH_cplx_Kx2(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k22 = dG_DH_cplx_Kx2(subdata['z'] - 1, subdata['z'] - 1, 1, subdata['r_A_red'] / AU_TO_ANG, subdata['r_AX_red'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    dG_k02 = (subdata['G_cplx_ox'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k02) * AU_TO_KJMOL
    dG_k12 = (subdata['G_cplx_rad'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k12) * AU_TO_KJMOL
    dG_k22 = (subdata['G_cplx_red'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k22) * AU_TO_KJMOL
    
    k02 = numpy.exp(-dG_k02 / (R * T))
    k12 = numpy.exp(-dG_k12 / (R * T))
    k22 = numpy.exp(-dG_k22 / (R * T))
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], numpy.log10(k02), 'o', color=color, label=family.replace('Family.', ''))
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], numpy.log10(k12), '^', color=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], numpy.log10(k22), 's', color=color)

def helpline_K02(ax, data: pandas.DataFrame, solvent: str, color: str = 'black'):
    subdata = data[data['solvent'] == solvent]
    
    dG_DH_k12 = dG_DH_cplx_Kx2(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_k12 = (subdata['G_cplx_rad'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k12) * AU_TO_KJMOL
    k12 = numpy.exp(-dG_k12 / (R * T))
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], numpy.log10(k12), '--', color=color, linewidth=0.75)

def make_table(f, data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    
    dG_DH_k02 = dG_DH_cplx_Kx2(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_A_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k12 = dG_DH_cplx_Kx2(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k22 = dG_DH_cplx_Kx2(subdata['z'] - 1, subdata['z'] - 1, 1, subdata['r_A_red'] / AU_TO_ANG, subdata['r_AX_red'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    subdata.insert(1, 'dG_cplx_ox', (subdata['G_cplx_ox'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k02) * AU_TO_KJMOL)
    subdata.insert(1, 'dG_cplx_rad', (subdata['G_cplx_rad'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k12) * AU_TO_KJMOL)
    subdata.insert(1, 'dG_cplx_red', (subdata['G_cplx_red'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k22) * AU_TO_KJMOL)
    
    f.write(format_longtable(
        subdata, 
        titles=['', '$a_{\\ce{NAC+}}$', '$\\Delta{G}_{cplx}^\\star$', '', '$a_{\\ce{NAC^.}}$', '$\\Delta{G}_{cplx}^\\star$', '', '$a_{\\ce{NAC-}}$', '$\\Delta{G}_{cplx}^\\star$'], 
        line_maker=lambda r: [
            r['name'].replace('mol_', ''), 
            '{:.2f}'.format(r['r_AX_ox']), 
            '{:.1f}'.format(r['dG_cplx_ox']), 
            '',
            '{:.2f}'.format(r['r_AX_rad']), 
            '{:.1f}'.format(r['dG_cplx_rad']), 
            '',
            '{:.2f}'.format(r['r_AX_red']), 
            '{:.1f}'.format(r['dG_cplx_red']), 
        ]
    ))
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_cplx_Kx2.csv')
parser.add_argument('-o', '--output', default='Data_cplx_Kx2.pdf')
parser.add_argument('-t', '--table')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(7, 8))
ax1, ax2 = figure.subplots(2, 1, sharey=True, sharex=True)

helpline_K02(ax1, data, 'water', 'black')

plot_Kx2(ax1, data, 'Family.AMO', 'water', 'tab:pink')
plot_Kx2(ax1, data, 'Family.P6O', 'water', 'tab:blue')
plot_Kx2(ax1, data, 'Family.P5O', 'water',  'black')
plot_Kx2(ax1, data, 'Family.IIO', 'water',  'tab:green')
plot_Kx2(ax1, data, 'Family.APO', 'water',  'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(38, -2, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')
ax1.plot([0, 62], [0, 0], '-', color='grey')

ax1.legend()

helpline_K02(ax2, data, 'acetonitrile', 'black')

plot_Kx2(ax2, data, 'Family.P6O', 'acetonitrile', 'tab:blue')
plot_Kx2(ax2, data, 'Family.P5O', 'acetonitrile',  'black')
plot_Kx2(ax2, data, 'Family.IIO', 'acetonitrile',  'tab:green')
plot_Kx2(ax2, data, 'Family.APO', 'acetonitrile',  'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
ax2.text(38, -2, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('log$_{10}$(K)') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, data, 'water')
        make_table(f, data, 'acetonitrile')
