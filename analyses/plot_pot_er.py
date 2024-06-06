import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import scipy
import argparse

from scipy.spatial import distance_matrix
from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV
from nitroxides.tex import format_longtable

LABELS = {'E_ox': [], 'E_red': []}
POINTS_POSITION ={'E_ox': [], 'E_red': []}
LABELS_KWARGS = {'E_ox': [], 'E_red': []}
LABELS_PATH = {'E_ox': pathlib.Path('pot_er_ox.pos'), 'E_red': pathlib.Path('pot_er_red.pos')}

AU_TO_DEBYE = 1 / 0.3934303 

EXCLUDE = [
    7, 19, 28, # ethyls
    9, 10, 20, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 55, # di-substituted
    56, 58, # ?!?
]

def plot_Er(ax, data: pandas.DataFrame, column: str, family: str, color: str):
    subdata = data[(data['solvent'] == 'water') &  (data['family'] == family) & data['px'].notnull()]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_ANG,  subdata['r_red'] / AU_TO_ANG, 80, 0) * AU_TO_EV
    
    Er = subdata['px'] / subdata['r'] ** 2 + subdata['Qxx'] / subdata['r'] ** 3
    
    excluded_ = [int(x.replace('mol_', '')) in EXCLUDE for x in subdata['name']]
    not_excluded_ = [not x for x in excluded_]
    
    ax.plot(Er[not_excluded_], subdata[not_excluded_][column]  - dG_DH_[not_excluded_], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(Er[excluded_], subdata[excluded_][column]  - dG_DH_[excluded_], '^', color=color)

    for name, er, e in zip(subdata['name'], Er, subdata[column]  - dG_DH_):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((er, e))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

def plot_corr_Er(ax, data: pandas.DataFrame, column: str):
    subdata = data[(data['solvent'] == 'water') & data['px'].notnull()]
    subdata = subdata[[int(x.replace('mol_', '')) not in EXCLUDE for x in subdata['name']]]
    
    if column == 'E_ox':
        dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad']  / AU_TO_ANG, 80, 0) * AU_TO_EV
    else:
        dG_DH_ = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_ANG,  subdata['r_red']  / AU_TO_ANG, 80, 0) * AU_TO_EV
    
    result = scipy.stats.linregress(subdata['px'] / subdata['r'] ** 2 + subdata['Qxx'] / subdata['r'] ** 3, subdata[column]  - dG_DH_)
    
    x = numpy.array([-.2, 1.5])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    ax.text(1.2,  1.2*result.slope+result.intercept+.05, '$R^2$={:.2f}'.format(result.rvalue **2))

def make_table(f, data: pandas.DataFrame, solvent: str):
    subdata = data[(data['solvent'] == solvent) & data['px'].notnull()]
    
    f.write(format_longtable(
        subdata,
        titles=['', '$r$', '$\\mu_x$', '$Q_{xx}$'], 
        line_maker=lambda r: [
            '{}'.format(int(r['name'].replace('mol_', ''))), 
            '{:.2f}'.format(r['r']), 
            '{:.2f}'.format(r['px']  / AU_TO_DEBYE), 
            '{:.2f}'.format(r['Qxx'] / AU_TO_DEBYE / AU_TO_ANG),
        ]
    ))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_er.pdf')
parser.add_argument('-t', '--table')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(7, 10))
ax3, ax4 = figure.subplots(2, 1)

plot_Er(ax3, data, 'E_ox', 'Family.P6O', 'tab:blue')
plot_Er(ax3, data, 'E_ox', 'Family.P5O', 'black')
plot_Er(ax3, data, 'E_ox', 'Family.IIO', 'tab:green')
plot_Er(ax3, data, 'E_ox', 'Family.APO', 'tab:red')
plot_corr_Er(ax3, data, 'E_ox')

ax3.legend()

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_ox'], 
    numpy.array(POINTS_POSITION['E_ox']), 
    LABELS['E_ox'], 
    labels_kwargs=LABELS_KWARGS['E_ox']
)

if args.reposition_labels: 
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=10000, c=0.02, b0=0.015, scale=(.2, 1))
    positioner.save(LABELS_PATH['E_ox'])

positioner.add_labels(ax3)

plot_Er(ax4, data, 'E_red', 'Family.P6O', 'tab:blue')
plot_Er(ax4, data, 'E_red', 'Family.P5O', 'black')
plot_Er(ax4, data, 'E_red', 'Family.IIO', 'tab:green')
plot_Er(ax4, data, 'E_red', 'Family.APO', 'tab:red')
plot_corr_Er(ax4, data, 'E_red')

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_red'], 
    numpy.array(POINTS_POSITION['E_red']), 
    LABELS['E_red'], 
    labels_kwargs=LABELS_KWARGS['E_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4,  krep=1, kspring=10000, c=0.02, b0=0.015, scale=(.2, 1))
    positioner.save(LABELS_PATH['E_red'])

positioner.add_labels(ax4)

[ax.set_xlabel('Electrostatic potential $\\mu_x/r^2 + Q_{xx}/r^3$ (D Å$^{-2}$)') for ax in [ax3, ax4]]
ax3.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ (V)')
ax4.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ (V)')

plt.tight_layout()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, data, 'water')
