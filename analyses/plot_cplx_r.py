import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import G_DH, AU_TO_M, AU_TO_EV, kappa2, G_NME4, G_BF4, RADII_BF4, RADII_NME4, C_NITROXIDE

def plot_r(ax, data: pandas.DataFrame, family: str, solvent: str, epsilon_r: float, color: str):
    subdata = data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_anion'] == True)].join(
        data[(data['family'] == family) & (data['solvent'] == solvent) & (data['has_cation'] == True)].set_index('name'),
        on='name', lsuffix='ox', rsuffix='red', how='inner'
    )
    
    ax.bar([int(x.replace('mol_', '')) for x in subdata['name']], subdata['d_OXox'] -subdata['d_OXred'], color=color, label=family.replace('Family.', ''))
    

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_cplx_Kx1.csv')
parser.add_argument('-o', '--output', default='Data_cplx_r.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 8))
ax1, ax2 = figure.subplots(2, 1)

plot_r(ax1, data, 'Family.AMO', 'water', 80.,'tab:pink')
plot_r(ax1, data, 'Family.P6O', 'water', 80.,'tab:blue')
plot_r(ax1, data, 'Family.P5O', 'water', 80., 'black')
plot_r(ax1, data, 'Family.IIO', 'water', 80., 'tab:green')
plot_r(ax1, data, 'Family.APO', 'water', 80., 'tab:red')


ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
# ax1.text(38, 1, "Water", fontsize=18)
ax1.plot([0, 62], [0, 0], '-', color='grey')

plot_r(ax2, data, 'Family.P6O', 'acetonitrile', 35.,'tab:blue')
plot_r(ax2, data, 'Family.P5O', 'acetonitrile', 35., 'black')
plot_r(ax2, data, 'Family.IIO', 'acetonitrile', 35., 'tab:green')
plot_r(ax2, data, 'Family.APO', 'acetonitrile', 35., 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_xlim(0.5,61.5)
#ax2.text(38, -2, "Acetonitrile", fontsize=18)
ax2.plot([0, 62], [0, 0], '-', color='grey')

[ax.set_ylabel('$d_{ox} - d_{red}$ (Å)') for ax in [ax1, ax2]]

plt.tight_layout()
figure.savefig(args.output)
