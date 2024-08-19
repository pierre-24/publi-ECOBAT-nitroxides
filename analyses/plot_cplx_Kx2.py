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

def prepare_data(data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    
    dG_DH_k02 = dG_DH_cplx_Kx2(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_A_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k12 = dG_DH_cplx_Kx2(subdata['z'], subdata['z'], 1, subdata['r_A_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    dG_DH_k22 = dG_DH_cplx_Kx2(subdata['z'] - 1, subdata['z'] - 1, 1, subdata['r_A_red'] / AU_TO_ANG, subdata['r_AX_red'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent])
    
    dG_k02 = (subdata['G_cplx_ox'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k02) * AU_TO_KJMOL
    dG_k12 = (subdata['G_cplx_rad'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k12) * AU_TO_KJMOL
    dG_k22 = (subdata['G_cplx_red'] - G_NME4[solvent] - G_BF4[solvent] + dG_DH_k22) * AU_TO_KJMOL
    
    subdata.insert(1, 'dG_cplx_ox', dG_k02)
    subdata.insert(1, 'dG_cplx_rad', dG_k12)
    subdata.insert(1, 'dG_cplx_red', dG_k22)
    
    subdata.insert(1, 'k02', numpy.exp(-dG_k02 / (R * T)))
    subdata.insert(1, 'k12', numpy.exp(-dG_k12 / (R * T)))
    subdata.insert(1, 'k22', numpy.exp(-dG_k22 / (R * T)))
    
    return subdata

def plot_Kx2(ax, data: pandas.DataFrame, family: str, color: str):
    subdata = data[data['family'] == family]
    
    x = [int(x.replace('mol_', '')) for x in subdata['name']]
    
    pK02 = -numpy.log10(subdata['k02'])
    pK12 = -numpy.log10(subdata['k12'])
    pK22 = -numpy.log10(subdata['k22'])
    
    ax.plot(x, -numpy.log10(subdata['k02']), 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(x, -numpy.log10(subdata['k12']), '^', color=color)
    ax.plot(x, -numpy.log10(subdata['k22']), 's', color=color)
    
    print('{} & {:.2f} $\\pm$ {:.2f} & {:.2f} $\\pm$ {:.2f} & {:.2f} $\\pm$ {:.2f} \\\\'.format(family, numpy.mean(pK02), numpy.std(pK02), numpy.mean(pK12), numpy.std(pK12), numpy.mean(pK22), numpy.std(pK22)))
    

def plot_helpline(ax, data):
    x = [int(x.replace('mol_', '')) for x in data['name']]
    ax.plot(x, -numpy.log10(data['k02']), '--', color='black', linewidth=0.8)

def make_table(f, data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    
    f.write(format_longtable(
        subdata, 
        titles=['', '$a_{\\ce{NAC+}}$', '$\\Delta{G}_{cplx}^\\star$', '', '$a_{\\ce{NAC^.}}$', '$\\Delta{G}_{cplx}^\\star$', '', '$a_{\\ce{NAC-}}$', '$\\Delta{G}_{cplx}^\\star$'], 
        line_maker=lambda r: [
            '{}'.format(int(r['name'].replace('mol_', ''))), 
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

subdata_wa = prepare_data(data, 'water')
plot_helpline(ax1, subdata_wa)

plot_Kx2(ax1, subdata_wa, 'Family.AMO', 'tab:pink')
plot_Kx2(ax1, subdata_wa, 'Family.P6O', 'tab:blue')
plot_Kx2(ax1, subdata_wa, 'Family.P5O', 'black')
plot_Kx2(ax1, subdata_wa, 'Family.IIO', 'tab:green')
plot_Kx2(ax1, subdata_wa, 'Family.APO', 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.text(38, 2, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')
ax1.plot([0, 62], [0, 0], '-', color='grey')

ax1.legend(ncols=5)

subdata_ac = prepare_data(data, 'acetonitrile')
plot_helpline(ax2, subdata_ac)

plot_Kx2(ax2, subdata_ac, 'Family.P6O', 'tab:blue')
plot_Kx2(ax2, subdata_ac, 'Family.P5O', 'black')
plot_Kx2(ax2, subdata_ac, 'Family.IIO', 'tab:green')
plot_Kx2(ax2, subdata_ac, 'Family.APO', 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
ax2.text(38, 2, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('pK$_{x2}$') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, subdata_wa, 'water')
        make_table(f, subdata_ac, 'acetonitrile')
