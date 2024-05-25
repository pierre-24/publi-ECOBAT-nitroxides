import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import argparse
from scipy.spatial import distance_matrix
from matplotlib.patches import Ellipse
from nitroxides.commons import dG_DH, AU_TO_M, LabelPositioner

SOLVENT = 'water'

LABELS = []
POINTS_POSITION = []
LABELS_KWARGS = []
LABELS_PATH = pathlib.Path('pot_{}.pos'.format(SOLVENT))

def plot_family(ax, data: pandas.DataFrame, family: str, color: str):
    subdata = data[numpy.logical_and(data['family'] == family, data['solvent'] == SOLVENT)]
    epsilon_r = 80 if SOLVENT == 'water' else 35
    
    dG_DH_ox0 = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_M * 1e-10, subdata['r_rad'] / AU_TO_M * 1e-10, epsilon_r, 0)
    dG_DH_red0 = dG_DH(subdata['z'], subdata['z'] - 1, subdata['r_rad'] / AU_TO_M * 1e-10,  subdata['r_red'] / AU_TO_M * 1e-10, epsilon_r, 0)
    
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

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_{}.pdf'.format(SOLVENT))

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
