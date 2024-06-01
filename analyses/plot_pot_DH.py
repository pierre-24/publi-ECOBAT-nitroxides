import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import argparse

from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from nitroxides.commons import dG_DH, AU_TO_M, AU_TO_EV

def plot_DH(ax, data: pandas.DataFrame, family: str, solvent: str, epsilon_r: float, c_elt: float, color: str):
    subdata = data[numpy.logical_and(data['family'] == family, data['solvent'] == solvent)]
    
    dG_DH_ox0 = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, epsilon_r, 0) * AU_TO_EV
    dG_DH_red0 = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, epsilon_r, 0) * AU_TO_EV
    dG_DH_ox01 = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, epsilon_r, 0.1) * AU_TO_EV
    dG_DH_red01 = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, epsilon_r, 0.1) * AU_TO_EV
    dG_DH_ox1 = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, epsilon_r, c_elt) * AU_TO_EV
    dG_DH_red1 = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, epsilon_r, c_elt) * AU_TO_EV
    
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_ox1, 'o', color=color, label=family.replace('Family.', ''))
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_red1, 'o', fillstyle='none', mec=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_ox01, 'v', color=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_red01, '^', fillstyle='none', mec=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_ox0, 's', color=color)
    ax.plot([int(x.replace('mol_', '')) for x in subdata['name']], -dG_DH_red0, 's', fillstyle='none', mec=color)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-o', '--output', default='Data_pot_DH.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 10))
ax1, ax2 = figure.subplots(2, 1, sharey=True)

plot_DH(ax1, data, 'Family.AMO', 'water', 80., 1, 'tab:pink')
plot_DH(ax1, data, 'Family.P6O', 'water', 80., 1, 'tab:blue')
plot_DH(ax1, data, 'Family.P5O', 'water', 80., 1, 'black')
plot_DH(ax1, data, 'Family.IIO', 'water', 80., 1, 'tab:green')
plot_DH(ax1, data, 'Family.APO', 'water', 80., 1, 'tab:red')

ax1.legend(ncols=5)
ax1.set_xlim(0.5,61.5)
ax1.plot([0, 62], [0, 0], '-', color='grey')
ax1.text(38, -0.03, "Water", fontsize=18)
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.grid(which='both', axis='x')

ax1.set_ylabel('$\\Delta E^0$ (V)') 

plot_DH(ax2, data, 'Family.P6O', 'acetonitrile', 35., 1, 'tab:blue')
plot_DH(ax2, data, 'Family.P5O', 'acetonitrile', 35., 1, 'black')
plot_DH(ax2, data, 'Family.IIO', 'acetonitrile', 35., 1, 'tab:green')
plot_DH(ax2, data, 'Family.APO', 'acetonitrile', 35., 1, 'tab:red')

ax2.set_xlabel('Molecule id') 
ax2.set_ylabel('$\\Delta E^0$ (V)') 
ax2.set_xlim(0.5,61.5)
ax2.plot([0, 62], [0, 0], '-', color='grey')
ax2.text(38, -0.03, "Acetonitrile", fontsize=18)
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax2.grid(which='both', axis='x')

plt.tight_layout()
figure.savefig(args.output)
