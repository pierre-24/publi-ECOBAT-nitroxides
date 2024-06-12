import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse
import scipy

from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, AU_TO_KJMOL, EPSILON_R, E_SHE, G_NME4, G_BF4, RADII_BF4, RADII_NME4, dG_DH_cplx_Kx1, EPSILON_R

LABELS = {'water': [], 'acetonitrile': []}
POINTS_POSITION ={'water': [], 'acetonitrile': []}
LABELS_KWARGS = {'water': [], 'acetonitrile': []}
LABELS_PATH = {'water': pathlib.Path('pot_exp_water.pos'), 'acetonitrile': pathlib.Path('pot_exp_acetonitrile.pos')}

EXCLUDE = [57, 51, 59]

T = 298.15
R = 8.3145e-3 # kJ mol⁻¹
F =  9.64853321233100184e4  # C mol⁻¹


def Ef_ox(E0: float,X: float, k01: float, k02: float, k11: float, k12: float):
    return E0 + R * T / F * numpy.log((1+k11*X+k12*X**2) / (1+k01*X+k02*X**2))


def prepare_data(data: pandas.DataFrame, data_exp: pandas.DataFrame, data_Kx1: pandas.DataFrame, solvent, correct: bool = True):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_exp[data_exp['E_ox_exp_{}'.format(solvent)].notnull()].set_index('compound'), on='compound', how='inner')
    
    subdata_Kx1 = data_Kx1[data_Kx1['solvent'] == solvent]
    subdata = subdata.join(subdata_Kx1[['name', 'r_AX_ox', 'r_AX_rad', 'G_cplx_ox', 'G_cplx_rad']].set_index('name'), on='name', how='inner')
    
    dG_DH_k01 = dG_DH_cplx_Kx1(subdata['z'] + 1, subdata['z'] + 1, 1, subdata['r_ox'] / AU_TO_ANG, subdata['r_AX_ox'] / AU_TO_ANG, RADII_BF4[solvent] / AU_TO_ANG, EPSILON_R[solvent], c_elt=0.1)
    dG_DH_k11 = dG_DH_cplx_Kx1(subdata['z'], subdata['z'], 1, subdata['r_rad'] / AU_TO_ANG, subdata['r_AX_rad'] / AU_TO_ANG, RADII_NME4[solvent] / AU_TO_ANG, EPSILON_R[solvent], c_elt=0.1)
    
    dG_k01 = (subdata['G_cplx_ox'] - G_BF4[solvent] + dG_DH_k01) * AU_TO_KJMOL
    dG_k11 = (subdata['G_cplx_rad'] - G_NME4[solvent] + dG_DH_k11) * AU_TO_KJMOL
    
    subdata.insert(1, 'dG_cplx_ox', dG_k01)
    subdata.insert(1, 'dG_cplx_rad', dG_k11)
    
    subdata.insert(1, 'k01', numpy.exp(-dG_k01 / (R * T)))
    subdata.insert(1, 'k11', numpy.exp(-dG_k11 / (R * T)))
    
    subdata['E_ox_exp_{}'.format(solvent)] /= 1000
    
    dG_DH_ = dG_DH(subdata['z'] + 1, subdata['z'], subdata['r_ox'] / AU_TO_ANG, subdata['r_rad'] / AU_TO_ANG, EPSILON_R[solvent], c_elt=0.1) * AU_TO_EV
    
    if correct:
        subdata.insert(1, 'E_ox_theo_{}'.format(solvent), subdata['E_ox'] - dG_DH_ - E_SHE[solvent])
        # subdata.insert(1, 'E_ox_theo_{}'.format(solvent), Ef_ox(subdata['E_ox'] - dG_DH_, 0.1, subdata['k01'], 0, subdata['k11'], 0) - E_SHE[solvent]) → some data are missing for the moment
    else:
        subdata.insert(1, 'E_ox_theo_{}'.format(solvent), subdata['E_ox'] - E_SHE[solvent])
    
    return subdata[['compound', 'family', 'E_ox_theo_{}'.format(solvent), 'E_ox_exp_{}'.format(solvent), 'k01', 'k11']]
    

def plot_exp_vs_theo(ax, data: pandas.DataFrame, solvent: str, family: str, color: str):
    subdata = data[data['family'] == family]
    
    e = EXCLUDE.copy()
    
    if solvent == 'acetonitrile':
        e += [4]
    
    ax.plot(subdata[~subdata['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], subdata[~subdata['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(subdata[subdata['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], subdata[subdata['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)], '^', color=color)
    
    for name, etheo, eexp in zip(subdata['compound'], subdata['E_ox_theo_{}'.format(solvent)], subdata['E_ox_exp_{}'.format(solvent)]):
        LABELS[solvent].append(name)
        POINTS_POSITION[solvent].append((etheo, eexp))
        LABELS_KWARGS[solvent].append(dict(color=color, ha='center', va='center'))

def plot_corr(ax, data: pandas.DataFrame, solvent: str):
    e = EXCLUDE.copy()
    
    if solvent == 'acetonitrile':
        e += [4]
    
    x, y = data[~data['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], data[~data['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)]
    result = scipy.stats.linregress(x, y)
    
    mae = numpy.mean(numpy.abs(x-y))
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = x.min()
    ax.text(x + .05, result.slope*x + result.intercept, '{:.2f} $\\times E^f_{{rel}}$ + {:.2f} V\n($R^2$={:.2f}, MAE={:.2f} V)'.format(result.slope, result.intercept,result.rvalue **2, mae))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-i2', '--input2', default='../data/Data_pot_ox_exp.csv')
parser.add_argument('-i3', '--input3', default='../data/Data_cplx_Kx1.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_ox_exp.pdf')
parser.add_argument('-R', '--raw', action='store_true')

args = parser.parse_args()

data = pandas.read_csv(args.input)
data_exp = pandas.read_csv(args.input2)
data_Kx1 = pandas.read_csv(args.input3)

figure = plt.figure(figsize=(5, 9))
ax1, ax2 = figure.subplots(2, 1)

subdata_wa = prepare_data(data, data_exp, data_Kx1, 'water', correct=not args.raw)

plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.P6O', 'tab:blue')
plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.P5O', 'black')
plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.IIO', 'tab:green')
plot_exp_vs_theo(ax1, subdata_wa, 'water', 'Family.APO', 'tab:red')

plot_corr(ax1, subdata_wa, 'water')

ax1.legend()
ax1.text(0.85, 0.95, 'Water', fontsize=18)

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

subdata_ac = prepare_data(data, data_exp, data_Kx1, 'acetonitrile', correct=not args.raw)

plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.P6O', 'tab:blue')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.P5O', 'black')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.IIO', 'tab:green')
plot_exp_vs_theo(ax2, subdata_ac, 'acetonitrile', 'Family.APO', 'tab:red')

plot_corr(ax2, subdata_ac, 'acetonitrile')
ax2.text(0.5, 1.15, 'Acetonitrile', fontsize=18)

positioner = LabelPositioner.from_file(
    LABELS_PATH['acetonitrile'], 
    numpy.array(POINTS_POSITION['acetonitrile']), 
    LABELS['acetonitrile'], 
    labels_kwargs=LABELS_KWARGS['acetonitrile']
)

if args.reposition_labels:
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=1000, c=0.05, b0=0.015, scale=[0.5, 1])
    positioner.save(LABELS_PATH['acetonitrile'])

positioner.add_labels(ax2)
ax2.xaxis.set_major_formatter('{x:.2f}')
[ax.set_xlabel('Computed $E^f_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]
[ax.set_ylabel('Experimental $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]

plt.tight_layout()
figure.savefig(args.output)
