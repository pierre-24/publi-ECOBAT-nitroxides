import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import scipy
import argparse

from scipy.spatial import distance_matrix
from nitroxides.commons import dG_DH, AU_TO_M, LabelPositioner, AU_TO_EV

LABELS = {'E_ox': [], 'E_red': []}
POINTS_POSITION ={'E_ox': [], 'E_red': []}
LABELS_KWARGS = {'E_ox': [], 'E_red': []}
LABELS_PATH = {'E_ox': pathlib.Path('pot_hammet_ox.pos'), 'E_red': pathlib.Path('pot_hammet_red.pos')}

def plot_hammet(ax, data: pandas.DataFrame, column: str, family: str, color: str):
    subdata = data[(data['solvent'] == 'water') & (data['family'] == family) & data['hammet'].notnull()]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    
    ax.plot(subdata['hammet'], subdata[column]  - dG_DH_, 'o', color=color, label=family.replace('Family.', ''))
    
    for name, hammet, e in zip(subdata['name'], subdata['hammet'], subdata[column]  - dG_DH_):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((hammet, e))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

def plot_corr_hammet(ax, data: pandas.DataFrame, column: str):
    subdata = data[(data['solvent'] == 'water') & data['hammet'].notnull()]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
        
    result = scipy.stats.linregress(subdata['hammet'], subdata[column]  - dG_DH_)
    
    x = numpy.array([-1., 1.])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    ax.text(-.7, -result.slope + result.intercept, '$R^2$={:.3f}'.format(result.rvalue **2))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_hammet.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(9, 5))
ax1, ax2 = figure.subplots(1, 2)

plot_hammet(ax1, data, 'E_ox', 'Family.P6O', 'tab:blue')
plot_hammet(ax1, data, 'E_ox', 'Family.P5O', 'black')
plot_corr_hammet(ax1, data, 'E_ox')

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_ox'], 
    numpy.array(POINTS_POSITION['E_ox']), 
    LABELS['E_ox'], 
    labels_kwargs=LABELS_KWARGS['E_ox']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-4, beta=1e5, krep=1, kspring=1000, c=0.03, b0=0.01, scale=(.15, 1))
    positioner.save(LABELS_PATH['E_ox'])

positioner.add_labels(ax1)

plot_hammet(ax2, data, 'E_red', 'Family.P6O', 'tab:blue')
plot_hammet(ax2, data, 'E_red', 'Family.P5O', 'black')
plot_corr_hammet(ax2, data, 'E_red')

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_red'], 
    numpy.array(POINTS_POSITION['E_red']), 
    LABELS['E_red'], 
    labels_kwargs=LABELS_KWARGS['E_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4,  krep=1, kspring=1000, c=0.03, b0=0.015, scale=(.2, 1))
    positioner.save(LABELS_PATH['E_red'])

positioner.add_labels(ax2)

[ax.set_xlabel('Hammet constant $\\sigma$') for ax in [ax1, ax2]]
ax1.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ (V)') 
ax2.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ (V)') 

plt.tight_layout()
figure.savefig(args.output)
