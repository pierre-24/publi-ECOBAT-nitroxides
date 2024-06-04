import pandas
import matplotlib.pyplot as plt
import numpy
import sys
import pathlib
import argparse

from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, EPSILON_R, E_SHE

LABELS = {'water': [], 'acetonitrile': []}
POINTS_POSITION ={'water': [], 'acetonitrile': []}
LABELS_KWARGS = {'water': [], 'acetonitrile': []}
LABELS_PATH = {'water': pathlib.Path('pot_exp_water.pos'), 'acetonitrile': pathlib.Path('pot_exp_acetonitrile.pos')}

def prepare_data(data: pandas.DataFrame, data_exp: pandas.DataFrame, solvent):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_exp[data_exp['E_ox_exp_{}'.format(solvent)].notnull()].set_index('compound'), on='compound', how='inner')
    
    subdata['E_ox_exp_{}'.format(solvent)] /= 1000
    
    dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, EPSILON_R[solvent], 0.1) * AU_TO_EV
    subdata.insert(1, 'E_ox_theo_{}'.format(solvent), subdata['E_ox'] - dG_DH_ - E_SHE[solvent])
    
    return subdata[['compound', 'family', 'E_ox_theo_{}'.format(solvent), 'E_ox_exp_{}'.format(solvent)]]
    

def plot_solv(ax, data: pandas.DataFrame, solvent: str, family: str, color: str):
    subdata = data[data['family'] == family]
    ax.plot(subdata['E_ox_theo_{}'.format(solvent)], subdata['E_ox_exp_{}'.format(solvent)], 'o', color=color, label=family.replace('Family.', ''))
    
    for name, etheo, eexp in zip(subdata['compound'], subdata['E_ox_theo_{}'.format(solvent)], subdata['E_ox_exp_{}'.format(solvent)]):
        LABELS[solvent].append(name)
        POINTS_POSITION[solvent].append((etheo, eexp))
        LABELS_KWARGS[solvent].append(dict(color=color, ha='center', va='center'))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-i2', '--input2', default='../data/Data_pot_ox_exp.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_ox_exp.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)
data_exp = pandas.read_csv(args.input2)

figure = plt.figure(figsize=(10, 5))
ax1, ax2 = figure.subplots(1, 2)

subdata_wa = prepare_data(data, data_exp, 'water')

plot_solv(ax1, subdata_wa, 'water', 'Family.P6O', 'tab:blue')
plot_solv(ax1, subdata_wa, 'water', 'Family.P5O', 'black')
plot_solv(ax1, subdata_wa, 'water', 'Family.IIO', 'tab:green')
plot_solv(ax1, subdata_wa, 'water', 'Family.APO', 'tab:red')

positioner = LabelPositioner.from_file(
    LABELS_PATH['water'], 
    numpy.array(POINTS_POSITION['water']), 
    LABELS['water'], 
    labels_kwargs=LABELS_KWARGS['water']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e5, krep=1, kspring=1000, c=0.05, b0=0.01)
    positioner.save(LABELS_PATH['water'])

positioner.add_labels(ax1)

subdata_ac = prepare_data(data, data_exp, 'acetonitrile')

plot_solv(ax2, subdata_ac, 'acetonitrile', 'Family.P6O', 'tab:blue')
plot_solv(ax2, subdata_ac, 'acetonitrile', 'Family.P5O', 'black')
plot_solv(ax2, subdata_ac, 'acetonitrile', 'Family.IIO', 'tab:green')
plot_solv(ax2, subdata_ac, 'acetonitrile', 'Family.APO', 'tab:red')

positioner = LabelPositioner.from_file(
    LABELS_PATH['acetonitrile'], 
    numpy.array(POINTS_POSITION['acetonitrile']), 
    LABELS['acetonitrile'], 
    labels_kwargs=LABELS_KWARGS['acetonitrile']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=1000, c=0.05, b0=0.01, scale=[0.3, 1])
    positioner.save(LABELS_PATH['acetonitrile'])

positioner.add_labels(ax2)

ax2.legend()
[ax.set_xlabel('Computed $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]
[ax.set_ylabel('Experimental $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]

plt.tight_layout()
figure.savefig(args.output)
