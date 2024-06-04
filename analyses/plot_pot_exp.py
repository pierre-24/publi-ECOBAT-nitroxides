import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse
import scipy

from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, EPSILON_R, E_SHE

LABELS = {'water': [], 'acetonitrile': []}
POINTS_POSITION ={'water': [], 'acetonitrile': []}
LABELS_KWARGS = {'water': [], 'acetonitrile': []}
LABELS_PATH = {'water': pathlib.Path('pot_exp_water.pos'), 'acetonitrile': pathlib.Path('pot_exp_acetonitrile.pos')}

EXCLUDE = [57, 51, 59]

def prepare_data(data: pandas.DataFrame, data_exp: pandas.DataFrame, solvent):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_exp[data_exp['E_ox_exp_{}'.format(solvent)].notnull()].set_index('compound'), on='compound', how='inner')
    
    subdata['E_ox_exp_{}'.format(solvent)] /= 1000
    
    dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, EPSILON_R[solvent], 0.1) * AU_TO_EV
    subdata.insert(1, 'E_ox_theo_{}'.format(solvent), subdata['E_ox'] - dG_DH_ - E_SHE[solvent])
    
    return subdata[['compound', 'family', 'E_ox_theo_{}'.format(solvent), 'E_ox_exp_{}'.format(solvent)]]
    

def plot_exp_vs_theo(ax, data: pandas.DataFrame, solvent: str, family: str, color: str):
    subdata = data[data['family'] == family]
    
    ax.plot(subdata[~subdata['compound'].isin(EXCLUDE)]['E_ox_theo_{}'.format(solvent)], subdata[~subdata['compound'].isin(EXCLUDE)]['E_ox_exp_{}'.format(solvent)], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(subdata[subdata['compound'].isin(EXCLUDE)]['E_ox_theo_{}'.format(solvent)], subdata[subdata['compound'].isin(EXCLUDE)]['E_ox_exp_{}'.format(solvent)], '^', color=color)
    
    for name, etheo, eexp in zip(subdata['compound'], subdata['E_ox_theo_{}'.format(solvent)], subdata['E_ox_exp_{}'.format(solvent)]):
        LABELS[solvent].append(name)
        POINTS_POSITION[solvent].append((etheo, eexp))
        LABELS_KWARGS[solvent].append(dict(color=color, ha='center', va='center'))

def plot_corr(ax, data: pandas.DataFrame, solvent: str):
    x, y = data[~data['compound'].isin(EXCLUDE)]['E_ox_theo_{}'.format(solvent)], data[~data['compound'].isin(EXCLUDE)]['E_ox_exp_{}'.format(solvent)]
    result = scipy.stats.linregress(x, y)
    
    mae = numpy.mean(numpy.abs(x-y))
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = .95 * x.min()+ .05 * x.max()
    ax.text(x + .05, result.slope*x + result.intercept, '{:.2f} $\\times E^0_{{rel}}$ + {:.2f}\n($R^2$={:.2f}, MAE={:.2f} V)'.format(result.slope, result.intercept,result.rvalue **2, mae))

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

plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.P6O', 'tab:blue')
plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.P5O', 'black')

plot_corr(ax1, subdata_wa, 'water')

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

plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.P6O', 'tab:blue')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.P5O', 'black')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.IIO', 'tab:green')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.APO', 'tab:red')

plot_corr(ax2, subdata_ac, 'acetonitrile')

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
[ax.set_xlabel('Computed $E^f_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]
[ax.set_ylabel('Experimental $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]

plt.tight_layout()
figure.savefig(args.output)
