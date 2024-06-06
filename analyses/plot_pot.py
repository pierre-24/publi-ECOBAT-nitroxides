import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import argparse
from scipy.spatial import distance_matrix
from matplotlib.patches import Ellipse
from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, EPSILON_R
from nitroxides.tex import format_longtable

SOLVENT = 'water'

LABELS = []
POINTS_POSITION = []
LABELS_KWARGS = []
LABELS_PATH = pathlib.Path('pot_{}.pos'.format(SOLVENT))

def plot_family(ax, data: pandas.DataFrame, family: str, color: str):
    subdata = data[(data['family'] == family) & (data['solvent'] == SOLVENT)]
    
    dG_DH_ox0 = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, EPSILON_R[SOLVENT], 0) * AU_TO_EV
    dG_DH_red0 = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_ANG,  subdata['r_red'] / AU_TO_ANG, EPSILON_R[SOLVENT], 0) * AU_TO_EV
    
    ax.plot(subdata['E_ox'] - dG_DH_ox0, subdata['E_red'] - dG_DH_red0, 'o', color=color, label=family.replace('Family.', ''))
    
    for name, eox, ered in zip(subdata['name'], subdata['E_ox'] - dG_DH_ox0, subdata['E_red'] - dG_DH_red0):
        name = name.replace('mol_', '')
        LABELS.append(name)
        POINTS_POSITION.append((eox, ered))
        LABELS_KWARGS.append(dict(color=color, ha='center', va='center'))
        
    m_ox = numpy.mean(subdata['E_ox'] - dG_DH_ox0)
    m_red = numpy.mean(subdata['E_red'] - dG_DH_red0)
    std_ox = numpy.std(subdata['E_ox'] - dG_DH_ox0)
    std_red = numpy.std(subdata['E_red'] - dG_DH_red0)
    
    ax.add_artist(Ellipse((m_ox, m_red), 2 * std_ox, 2 * std_red, alpha=0.25, facecolor=color))

def make_table(f, data: pandas.DataFrame, solvent: str):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'dG_DH_ox', dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, EPSILON_R[SOLVENT], 0) * AU_TO_EV)
    subdata.insert(2, 'dG_DH_red', dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_ANG,  subdata['r_red'] / AU_TO_ANG, EPSILON_R[SOLVENT], 0) * AU_TO_EV)
    
    
    f.write(format_longtable(
        subdata, 
        titles=['', '$a_{\\ce{N+}}$', '$a_{\\ce{N^.}}$', '$a_{\\ce{N-}}$', '$E^0_{abs}(\\ce{N+}|\\ce{N^.})$', '$E^0_{abs}(\\ce{N^.}|\\ce{N-})$'], 
        line_maker=lambda r: [
            '{}'.format(int(r['name'].replace('mol_', ''))), 
            '{:.2f}'.format(r['r_ox']), 
            '{:.2f}'.format(r['r_rad']), 
            '{:.2f}'.format(r['r_red']),
            '{:.2f}'.format(r['E_ox'] - r['dG_DH_ox']), 
            '{:.2f}'.format(r['E_red'] - r['dG_DH_red']),
        ]
    ))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_{}.pdf'.format(SOLVENT))
parser.add_argument('-t', '--table')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(7, 6.5))
ax1 = figure.subplots(1, 1, sharey=True)

plot_family(ax1, data, 'Family.AMO', 'tab:pink')
plot_family(ax1, data, 'Family.P6O', 'tab:blue')
plot_family(ax1, data, 'Family.P5O', 'black')
plot_family(ax1, data, 'Family.IIO', 'tab:green')
plot_family(ax1, data, 'Family.APO', 'tab:red')

positioner = LabelPositioner.from_file(
    LABELS_PATH, 
    numpy.array(POINTS_POSITION), 
    LABELS, 
    labels_kwargs=LABELS_KWARGS
)

if args.reposition_labels:
    positioner.optimize(dx=1e-4, beta=1e7, krep=1, kspring=1000, c=0.05, b0=0.015)
    positioner.save(LABELS_PATH)

positioner.add_labels(ax1)

ax1.set_xlabel('$E^0_{abs}(N^+|N^\\bullet)$ (V)') 
ax1.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ (V)') 

plt.tight_layout()
plt.legend()
figure.savefig(args.output)

if args.table:
    with pathlib.Path(args.table).open('w') as f:
        make_table(f, data, 'water')
        make_table(f, data, 'acetonitrile')
