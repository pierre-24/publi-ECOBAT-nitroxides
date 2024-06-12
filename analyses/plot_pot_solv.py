import pandas
import matplotlib.pyplot as plt
import numpy
import scipy
import pathlib
import argparse

from nitroxides.commons import dG_DH, AU_TO_M, LabelPositioner, AU_TO_EV

LABELS = {'Ef_ox': [], 'Ef_red': []}
POINTS_POSITION ={'Ef_ox': [], 'Ef_red': []}
LABELS_KWARGS = {'Ef_ox': [], 'Ef_red': []}
LABELS_PATH = {'Ef_ox': pathlib.Path('pot_solv_ox.pos'), 'Ef_red': pathlib.Path('pot_solv_red.pos')}


def prepare_data(data: pandas.DataFrame):
    subdata = data[data['solvent'] == 'water'].join(data[data['solvent'] == 'acetonitrile'].set_index('name'), on='name', lsuffix='wa', rsuffix='ac', how='inner')
    
    dG_DH_ox_wa = dG_DH(subdata['zwa'] + 1, subdata['zwa'], subdata['r_oxwa'] / AU_TO_M * 1e-10, subdata['r_radwa'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    dG_DH_ox_ac = dG_DH(subdata['zac'] + 1, subdata['zac'], subdata['r_oxac'] / AU_TO_M * 1e-10, subdata['r_radac'] / AU_TO_M * 1e-10, 35, 0) * AU_TO_EV
    dG_DH_red_wa = dG_DH(subdata['zwa'], subdata['zwa'] - 1, subdata['r_radwa'] / AU_TO_M * 1e-10,  subdata['r_redwa'] / AU_TO_M * 1e-10, 80, 0) * AU_TO_EV
    dG_DH_red_ac = dG_DH(subdata['zac'], subdata['zac'] - 1, subdata['r_radac'] / AU_TO_M * 1e-10,  subdata['r_redac'] / AU_TO_M * 1e-10, 35, 0) * AU_TO_EV
    
    subdata.insert(1, 'Ef_oxwa', subdata['E_oxwa'] - dG_DH_ox_wa)
    subdata.insert(1, 'Ef_redwa', subdata['E_redwa'] - dG_DH_red_wa)
    subdata.insert(1, 'Ef_oxac', subdata['E_oxac'] - dG_DH_ox_ac)
    subdata.insert(1, 'Ef_redac', subdata['E_redac'] - dG_DH_red_ac)
    
    return subdata
    

def plot_solv(ax, data: pandas.DataFrame, column: str, family: str, color: str):
    subdata = data[data['familywa'] == family]
    
    ax.plot(subdata['{}wa'.format(column)], subdata['{}ac'.format(column)], 'o', color=color, label=family.replace('Family.', ''))
    
    for name, ewa, eac in zip(subdata['name'], subdata['{}wa'.format(column)], subdata['{}ac'.format(column)]):
        name = name.replace('mol_', '')
        LABELS[column].append(name)
        POINTS_POSITION[column].append((ewa, eac))
        LABELS_KWARGS[column].append(dict(color=color, ha='center', va='center'))

def plot_corr(ax, data: pandas.DataFrame, column: str):
        
    x, y = data['{}wa'.format(column)], data['{}ac'.format(column)]
    result = scipy.stats.linregress(x, y)
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = x.min()
    ax.text(x + .05, result.slope*x + result.intercept, '{:.2f} $\\times E^0_{{abs}}$ - {:.2f} V ($R^2$={:.2f})'.format(result.slope, -result.intercept, result.rvalue **2))


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_solv.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)
data = prepare_data(data)

figure = plt.figure(figsize=(5, 9))
ax1, ax2 = figure.subplots(2, 1)

plot_solv(ax1, data, 'Ef_ox', 'Family.P6O', 'tab:blue')
plot_solv(ax1, data, 'Ef_ox', 'Family.P5O', 'black')
plot_solv(ax1, data, 'Ef_ox', 'Family.IIO', 'tab:green')
plot_solv(ax1, data, 'Ef_ox', 'Family.APO', 'tab:red')

plot_corr(ax1, data, 'Ef_ox')

ax1.legend()
ax1.set_xlabel('$E^0_{abs}(N^+|N^\\bullet)$ in water (V)') 
ax1.set_ylabel('$E^0_{abs}(N^+|N^\\bullet)$ in acetonitrile (V)')

positioner = LabelPositioner.from_file(
    LABELS_PATH['Ef_ox'], 
    numpy.array(POINTS_POSITION['Ef_ox']), 
    LABELS['Ef_ox'], 
    labels_kwargs=LABELS_KWARGS['Ef_ox']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e5, krep=1, kspring=1000, c=0.05, b0=0.02)
    positioner.save(LABELS_PATH['Ef_ox'])

positioner.add_labels(ax1)

plot_solv(ax2, data, 'Ef_red', 'Family.P6O', 'tab:blue')
plot_solv(ax2, data, 'Ef_red', 'Family.P5O', 'black')
plot_solv(ax2, data, 'Ef_red', 'Family.IIO', 'tab:green')
plot_solv(ax2, data, 'Ef_red', 'Family.APO', 'tab:red')

plot_corr(ax2, data, 'Ef_red')

ax2.set_xlabel('$E^0_{abs}(N^\\bullet|N^-)$ in water (V)') 
ax2.set_ylabel('$E^0_{abs}(N^\\bullet|N^-)$ in acetonitrile (V)') 

positioner = LabelPositioner.from_file(
    LABELS_PATH['Ef_red'], 
    numpy.array(POINTS_POSITION['Ef_red']), 
    LABELS['Ef_red'], 
    labels_kwargs=LABELS_KWARGS['Ef_red']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-2, beta=1e4, krep=1, kspring=1000, c=0.1, b0=0.02, scale=[1, 0.7])
    positioner.save(LABELS_PATH['Ef_red'])

positioner.add_labels(ax2)

plt.tight_layout()
figure.savefig(args.output)
