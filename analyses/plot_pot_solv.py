import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import argparse

from nitroxides.commons import dG_DH, AU_TO_M, LabelPositioner, AU_TO_EV

LABELS = {'E_ox': [], 'E_red': []}
POINTS_POSITION ={'E_ox': [], 'E_red': []}
LABELS_KWARGS = {'E_ox': [], 'E_red': []}
LABELS_PATH = {'E_ox': pathlib.Path('pot_solv_ox.pos'), 'E_red': pathlib.Path('pot_solv_red.pos')}

def plot_solv(ax, data: pandas.DataFrame, column: str, family: str, color: str):
    subdata = data[numpy.logical_and(data['solvent'] == 'water', data['family'] == family)].join(data[numpy.logical_and(data['solvent'] == 'acetonitrile', data['family'] == family)].set_index('name'), on='name', lsuffix='wa', rsuffix='ac', how='inner')
    
    if column == 'E_ox':
        dG_DH_wa = dG_DH(subdata['zwa'] + 1, subdata['zwa'], subdata['r_oxwa'] / AU_TO_M * 1e-10, subdata['r_radwa'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
        dG_DH_ac = dG_DH(subdata['zac'] + 1, subdata['zac'], subdata['r_oxac'] / AU_TO_M * 1e-10, subdata['r_radac'] / AU_TO_M * 1e-10, 35, 0) * AU_TO_EV
    else:
        dG_DH_wa = dG_DH(subdata['zwa'], subdata['zwa'] - 1, subdata['r_radwa'] / AU_TO_M * 1e-10,  subdata['r_redwa'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
        dG_DH_ac = dG_DH(subdata['zac'], subdata['zac'] - 1, subdata['r_radac'] / AU_TO_M * 1e-10,  subdata['r_redac'] / AU_TO_M * 1e-10, 35, 0) * AU_TO_EV
    
    ax.plot(subdata['{}wa'.format(column)] - dG_DH_wa, subdata['{}ac'.format(column)] - dG_DH_ac, 'o', color=color, label=family.replace('Family.', ''))
    
    for name, ewa, eac in zip(subdata['name'], subdata['{}wa'.format(column)] - dG_DH_wa, subdata['{}ac'.format(column)] - dG_DH_ac):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((ewa, eac))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_solv.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)

figure = plt.figure(figsize=(10, 5))
ax1, ax2 = figure.subplots(1, 2)

plot_solv(ax1, data, 'E_ox', 'Family.P6O', 'tab:blue')
plot_solv(ax1, data, 'E_ox', 'Family.P5O', 'black')
plot_solv(ax1, data, 'E_ox', 'Family.IIO', 'tab:green')
plot_solv(ax1, data, 'E_ox', 'Family.APO', 'tab:red')

ax1.legend()
ax1.plot([5, 5.4], [5, 5.4], '--', color='black')
ax1.set_xlabel('$E^0_{abs}(N^+|N^\\bullet)$ in water (V)') 
ax1.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ in acetonitrile (V)')

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_ox'], 
    numpy.array(POINTS_POSITION['E_ox']), 
    LABELS['E_ox'], 
    labels_kwargs=LABELS_KWARGS['E_ox']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e6, krep=1, kspring=1000, c=0.05, b0=0.015)
    positioner.save(LABELS_PATH['E_ox'])

positioner.add_labels(ax1)

plot_solv(ax2, data, 'E_red', 'Family.P6O', 'tab:blue')
plot_solv(ax2, data, 'E_red', 'Family.P5O', 'black')
plot_solv(ax2, data, 'E_red', 'Family.IIO', 'tab:green')
plot_solv(ax2, data, 'E_red', 'Family.APO', 'tab:red')

ax2.plot([2, 3.4], [2, 3.4], '--', color='black')
ax2.set_xlabel('$E^0_{abs}(N^\\bullet|N^-)$ in water (V)') 
ax2.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ in acetonitrile (V)') 

positioner = LabelPositioner.from_file(
    LABELS_PATH['E_red'], 
    numpy.array(POINTS_POSITION['E_red']), 
    LABELS['E_red'], 
    labels_kwargs=LABELS_KWARGS['E_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=1000, c=0.3, b0=0.05)
    positioner.save(LABELS_PATH['E_red'])

positioner.add_labels(ax2)

plt.tight_layout()
figure.savefig(args.output)
