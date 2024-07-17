import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import scipy
import argparse

from scipy.spatial import distance_matrix
from nitroxides.commons import dG_DH, AU_TO_M, LabelPositioner, AU_TO_EV

LABELS = {'E_ox_hammet': [], 'E_red_hammet': [], 'E_ox_inductive': [], 'E_red_inductive': []}
POINTS_POSITION = {'E_ox_hammet': [], 'E_red_hammet': [], 'E_ox_inductive': [], 'E_red_inductive': []}
LABELS_KWARGS = {'E_ox_hammet': [], 'E_red_hammet': [], 'E_ox_inductive': [], 'E_red_inductive': []}
LABELS_PATH = {
    'E_ox_hammet': pathlib.Path('pot_hammet_ox.pos'), 
    'E_red_hammet': pathlib.Path('pot_hammet_red.pos'), 
    'E_ox_inductive': pathlib.Path('pot_hammet_ox_inductive.pos'), 
    'E_red_inductive': pathlib.Path('pot_hammet_red_inductive.pos')
}

def plot_hammet(ax, data: pandas.DataFrame, column: str, family: str, color: str, const: str = 'hammet'):
    subdata = data[(data['solvent'] == 'water') & (data['family'] == family) & data[const].notnull()]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    
    ax.plot(subdata[const], subdata[column]  - dG_DH_, 'o', color=color, label=family.replace('Family.', ''))
    
    for name, hammet, e in zip(subdata['name'], subdata[const], subdata[column]  - dG_DH_):
        name = name.replace('mol_', '')
        n = '{}_{}'.format(column, const)
        LABELS[n].append(name)
        POINTS_POSITION[n].append((hammet, e))
        LABELS_KWARGS[n].append(dict(color=color, ha='center', va='center'))

def plot_corr_hammet(ax, data: pandas.DataFrame, column: str, family: str, color: str, const: str = 'hammet'):
    subdata = data[(data['solvent'] == 'water') & (data['family'] == family) & data[const].notnull()]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
        
    result = scipy.stats.linregress(subdata[const], subdata[column]  - dG_DH_)
    
    x = numpy.array([-1., 1.])
    ax.plot(x, result.slope*x + result.intercept, '--', color=color)
    ax.text(-.9, -result.slope + result.intercept + .1, '$R^2$={:.2f}'.format(result.rvalue **2), color=color)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_hammet.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(8, 8))
(ax1, ax2), (ax3, ax4) = figure.subplots(2, 2)

for axes, const in [((ax1, ax2), 'hammet'), ((ax3, ax4), 'inductive'),]:
    
    for family, color in [('Family.P6O', 'tab:blue'), ('Family.P5O', 'black')]:
        plot_hammet(axes[0], data, 'E_ox', family, color, const=const)
        plot_corr_hammet(axes[0], data, 'E_ox', family, color, const=const)
    
    n = 'E_ox_{}'.format(const)

    positioner = LabelPositioner.from_file(
        LABELS_PATH[n], 
        numpy.array(POINTS_POSITION[n]), 
        LABELS[n], 
        labels_kwargs=LABELS_KWARGS[n]
    )

    if args.reposition_labels:
        positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=1000, c=0.03, b0=0.02, scale=(.2, 1))
        positioner.save(LABELS_PATH[n])

    positioner.add_labels(axes[0])
    
    for family, color in [('Family.P6O', 'tab:blue'), ('Family.P5O', 'black')]:
        plot_hammet(axes[1], data, 'E_red', family, color, const=const)
        plot_corr_hammet(axes[1], data, 'E_red', family, color, const=const)
    
    n = 'E_red_{}'.format(const)

    positioner = LabelPositioner.from_file(
        LABELS_PATH[n], 
        numpy.array(POINTS_POSITION[n]), 
        LABELS[n], 
        labels_kwargs=LABELS_KWARGS[n]
    )

    if args.reposition_labels:
        positioner.optimize(dx=1e-3, beta=1e4,  krep=1, kspring=1000, c=0.04, b0=0.03, scale=(.3, 1))
        positioner.save(LABELS_PATH[n])

    positioner.add_labels(axes[1])

[ax.set_xlabel('Hammet constant $\\sigma_m$ or $\\sigma_p$') for ax in [ax1, ax2]]
[ax.set_xlabel('Inductive constante $\\sigma_I$') for ax in [ax3, ax4]]
[ax.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ (V)') for ax in [ax1, ax3]]
[ax.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ (V)') for ax in [ax2, ax4]]

plt.tight_layout()
figure.savefig(args.output)
