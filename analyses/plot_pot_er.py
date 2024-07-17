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

LABELS = {'Ef_ox': [], 'Ef_red': []}
POINTS_POSITION ={'Ef_ox': [], 'Ef_red': []}
LABELS_KWARGS = {'Ef_ox': [], 'Ef_red': []}
LABELS_PATH = {'Ef_ox': pathlib.Path('pot_er_ox.pos'), 'Ef_red': pathlib.Path('pot_er_red.pos')}

AU_TO_DEBYE = 1 / 0.3934303 

EXCLUDE = [
    7, 19, 28, # ethyls
    9, 10, 20, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 55, # di-substituted
    56, 58, # ?!?
]

def prepare_data(data: pandas.DataFrame):
    subdata = data[(data['solvent'] == 'water') & data['px'].notnull()]
    
    dG_DH_ox = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, 80, 0) * AU_TO_EV
    dG_DH_red = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_ANG,  subdata['r_red'] / AU_TO_ANG, 80, 0) * AU_TO_EV
    
    subdata.insert(1, 'Ef_ox', subdata['E_ox'] - dG_DH_ox)
    subdata.insert(1, 'Ef_red', subdata['E_red'] - dG_DH_red)
    
    r =  subdata['r'] / AU_TO_ANG
    Er = (subdata['px'] / AU_TO_DEBYE / r ** 2 + subdata['Qxx'] / AU_TO_DEBYE / AU_TO_ANG / r ** 3) * AU_TO_EV
    subdata.insert(1, 'Er', Er)
    
    return subdata

def plot_Er(ax, data: pandas.DataFrame, column: str, family: str, color: str):
    subdata = data[data['family'] == family]
    
    excluded_ = [int(x.replace('mol_', '')) in EXCLUDE for x in subdata['name']]
    not_excluded_ = [not x for x in excluded_]
    
    f = 1 if column == 'Ef_ox' else -1
    ax.plot(f * subdata[not_excluded_]['Er'], subdata[not_excluded_][column], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(f * subdata[excluded_]['Er'], subdata[excluded_][column], '^', color=color)

    for name, er, e in zip(subdata['name'], f * subdata['Er'], subdata[column]):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((er, e))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

def plot_corr_Er(ax, data: pandas.DataFrame, column: str):
    subdata = data[[int(x.replace('mol_', '')) not in EXCLUDE for x in data['name']]]
    f = 1 if column == 'Ef_ox' else -1
    result = scipy.stats.linregress(f * subdata['Er'], subdata[column])
    
    x = numpy.array([(f * subdata['Er']).min(), (f * subdata['Er']).max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    ax.text(3 * f, result.intercept - .05, '$R^2$={:.2f}'.format(result.rvalue **2), size=12)

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
data = prepare_data(data)

figure = plt.figure(figsize=(7, 10))
ax3, ax4 = figure.subplots(2, 1)

plot_Er(ax3, data, 'Ef_ox', 'Family.P6O', 'tab:blue')
plot_Er(ax3, data, 'Ef_ox', 'Family.P5O', 'black')
plot_Er(ax3, data, 'Ef_ox', 'Family.IIO', 'tab:green')
plot_Er(ax3, data, 'Ef_ox', 'Family.APO', 'tab:red')
plot_corr_Er(ax3, data, 'Ef_ox')

ax3.legend()

positioner = LabelPositioner.from_file(
    LABELS_PATH['Ef_ox'], 
    numpy.array(POINTS_POSITION['Ef_ox']), 
    LABELS['Ef_ox'], 
    labels_kwargs=LABELS_KWARGS['Ef_ox']
)

if args.reposition_labels: 
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=10000, c=0.02, b0=0.015, scale=(.1, 1))
    positioner.save(LABELS_PATH['Ef_ox'])

positioner.add_labels(ax3)

plot_Er(ax4, data, 'Ef_red', 'Family.P6O', 'tab:blue')
plot_Er(ax4, data, 'Ef_red', 'Family.P5O', 'black')
plot_Er(ax4, data, 'Ef_red', 'Family.IIO', 'tab:green')
plot_Er(ax4, data, 'Ef_red', 'Family.APO', 'tab:red')
plot_corr_Er(ax4, data, 'Ef_red')

positioner = LabelPositioner.from_file(
    LABELS_PATH['Ef_red'], 
    numpy.array(POINTS_POSITION['Ef_red']), 
    LABELS['Ef_red'], 
    labels_kwargs=LABELS_KWARGS['Ef_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4,  krep=1, kspring=10000, c=0.02, b0=0.015, scale=(.1, 1))
    positioner.save(LABELS_PATH['Ef_red'])

positioner.add_labels(ax4)

[ax.set_xlabel('Electrostatic potential $U_q$ (V)') for ax in [ax3, ax4]]
ax3.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ (V)')
ax4.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ (V)')

plt.tight_layout()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, data, 'water')
