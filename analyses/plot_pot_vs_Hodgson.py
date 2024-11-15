import pandas
import matplotlib.pyplot as plt
import numpy
import scipy
import pathlib
import argparse
from scipy.spatial import distance_matrix
from matplotlib.patches import Ellipse
from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, EPSILON_R
from nitroxides.tex import format_longtable

SOLVENT = 'water'

LABELS = {'E_ox': [], 'E_red': []}
POINTS_POSITION ={'E_ox': [], 'E_red': []}
LABELS_KWARGS = {'E_ox': [], 'E_red': []}
LABELS_PATH = {'E_ox': pathlib.Path('pot_hodgson_ox.pos'), 'E_red': pathlib.Path('pot_hodgson_red.pos')}

TO_SHE = 4.36  # Eq. 4 of Hodgson et al.

def prepare_data(data: pandas.DataFrame, data_hog: pandas.DataFrame):
    subdata = data[data['solvent'] == SOLVENT]
    
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_hog[data_hog['E_red'].notnull()].set_index('compound'), on='compound', lsuffix='our', rsuffix='them', how='inner')
    
    return subdata
    
def plot_family(ax, data: pandas.DataFrame, family: str, column: str, color: str, e = []):
    
    data = data[data['family'] == family]
    ax.plot(data[~data['compound'].isin(e)][column + 'our'], data[~data['compound'].isin(e)][column + 'them'] / 1000 + TO_SHE, 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(data[data['compound'].isin(e)][column + 'our'], data[data['compound'].isin(e)][column + 'them'] / 1000 + TO_SHE, '^', color=color)
    
    for name, e1, e2 in zip(data['name'], data[column + 'our'], data[column + 'them'] / 1000 + TO_SHE):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((e1, e2))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

def plot_corr(ax, data: pandas.DataFrame, column: str, e= []):
        
    x, y = data[~data['compound'].isin(e)][column + 'our'], data[~data['compound'].isin(e)][column + 'them'] / 1000 + TO_SHE
    result = scipy.stats.linregress(x, y)
    mae = numpy.mean(numpy.abs(x-y))
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = x.min()
    ax.text(x + .05, result.slope*x + result.intercept - 0.15, '{:.2f} $\\times E^0_{{abs}}$ {:+.2f} V\n($R^2$={:.2f}, MAE={:.2f} V)'.format(result.slope, result.intercept, result.rvalue **2, mae))


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-i2', '--input2', default='../data/Data_pot_Hodgson.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_vs_Hodgson.pdf')

args = parser.parse_args()

data_our = pandas.read_csv(args.input)
data_hog = pandas.read_csv(args.input2)

data = prepare_data(data_our, data_hog)

figure = plt.figure(figsize=(6, 10))
ax1, ax2 = figure.subplots(2, 1)

plot_family(ax1, data, 'Family.AMO', 'E_ox', 'tab:pink')
plot_family(ax1, data, 'Family.P6O', 'E_ox', 'tab:blue')
plot_family(ax1, data,  'Family.P5O', 'E_ox', 'black')
plot_family(ax1, data, 'Family.IIO', 'E_ox', 'tab:green')
plot_family(ax1, data, 'Family.APO', 'E_ox', 'tab:red', list(range(36, 56)))

plot_corr(ax1, data, 'E_ox', list(range(36, 56)))

ax1.set_xlabel('$E^0_{abs}(N^+|N^\\bullet)$ from this work (V)') 
ax1.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ from Hodgson et al. (V)') 
ax1.legend()

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_ox'], 
    numpy.array(POINTS_POSITION['E_ox']), 
    LABELS['E_ox'], 
    labels_kwargs=LABELS_KWARGS['E_ox']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e5, krep=1, kspring=1000, c=0.05, b0=0.015)
    positioner.save(LABELS_PATH['E_ox'])

positioner.add_labels(ax1)

plot_family(ax2, data, 'Family.AMO', 'E_red', 'tab:pink')
plot_family(ax2, data, 'Family.P6O', 'E_red', 'tab:blue')
plot_family(ax2, data, 'Family.P5O', 'E_red', 'black')
plot_family(ax2, data, 'Family.IIO', 'E_red', 'tab:green', [30])
plot_family(ax2, data, 'Family.APO', 'E_red', 'tab:red', [53, 54])

plot_corr(ax2, data, 'E_red',  [30, 53, 54])

ax2.set_xlabel('$E^0_{abs}(N^\\bullet|N^-)$ from this work (V)') 
ax2.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ from Hodgson et al. (V)') 
ax2.xaxis.set_major_formatter('{x:.2f}')

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_red'], 
    numpy.array(POINTS_POSITION['E_red']), 
    LABELS['E_red'], 
    labels_kwargs=LABELS_KWARGS['E_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e3, krep=1, kspring=1000, c=0.3, b0=0.05, scale=[3, 1])
    positioner.save(LABELS_PATH['E_red'])

positioner.add_labels(ax2)

plt.tight_layout()
figure.savefig(args.output)
