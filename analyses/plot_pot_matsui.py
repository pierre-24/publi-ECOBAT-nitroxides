import pandas
import matplotlib.pyplot as plt
import numpy
import pathlib
import argparse
import scipy

from nitroxides.commons import dG_DH, AU_TO_ANG, LabelPositioner, AU_TO_EV, EPSILON_R, E_SHE as E_SHE_

LABELS = {'water': [], 'acetonitrile': []}
POINTS_POSITION ={'water': [], 'acetonitrile': []}
LABELS_KWARGS = {'water': [], 'acetonitrile': []}
LABELS_PATH = {'water': pathlib.Path('pot_matsui_water.pos'), 'acetonitrile': pathlib.Path('pot_matsui_acetonitrile.pos')}

EXCLUDE = [57, 51, 59]

def matsui_E_P_rel(E_abs, E_SHE, f, mu, a0, epsilon_r, z, n_e=1):
    return E_abs - 1/(2*numpy.abs(f)*a0 / AU_TO_ANG) * (1/epsilon_r-1)*((n_e-z)**2*scipy.special.erf(mu*a0*numpy.abs(n_e-z))-z**2*scipy.special.erf(mu*a0*numpy.abs(z))) * 27.212 - E_SHE # in V
    

INIT_FIT = {'water': [4.34, 0.7, 1e-3], 'acetonitrile': [4.26, 0.6, 1e-2]}

def prepare_data(data: pandas.DataFrame, data_exp: pandas.DataFrame, solvent):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_exp[data_exp['E_ox_exp_{}'.format(solvent)].notnull()].set_index('compound'), on='compound', how='inner')
    
    e = EXCLUDE.copy()
    
    if solvent == 'acetonitrile':
        e += [12, 4]
    
    subdata['E_ox_exp_{}'.format(solvent)] /= 1000
    
    def matsui(x, E_SHE, f, mu):
        return matsui_E_P_rel(x[~subdata['compound'].isin(e)], E_SHE, f, mu, subdata[~subdata['compound'].isin(e)]['r_rad'], EPSILON_R[solvent], subdata[~subdata['compound'].isin(e)]['z'] + 1)
    
    (mat_E_SHE, mat_f, mat_mu), cov = scipy.optimize.curve_fit(
        matsui, 
        subdata[~subdata['compound'].isin(e)]['E_ox'], 
        subdata[~subdata['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)],
        p0=INIT_FIT[solvent],
        method='dogbox'
    )
    
    subdata.insert(1, 'E_ox_theo_{}'.format(solvent), matsui_E_P_rel(subdata['E_ox'], mat_E_SHE, mat_f, mat_mu, subdata['r_rad'], EPSILON_R[solvent], subdata['z'] + 1))
    
    return subdata[['compound', 'family', 'E_ox_theo_{}'.format(solvent), 'E_ox_exp_{}'.format(solvent)]], (mat_E_SHE, mat_f, numpy.abs(mat_mu))
    

def plot_exp_vs_matsui(ax, data: pandas.DataFrame, solvent: str, family: str, color: str):
    subdata = data[data['family'] == family]
    e = EXCLUDE.copy()
    
    if solvent == 'acetonitrile':
        e += [12, 4]
    
    ax.plot(subdata[~subdata['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], subdata[~subdata['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)], 'o', color=color, label=family.replace('Family.', ''))
    ax.plot(subdata[subdata['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], subdata[subdata['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)], '^', color=color)
    
    for name, etheo, eexp in zip(subdata['compound'], subdata['E_ox_theo_{}'.format(solvent)], subdata['E_ox_exp_{}'.format(solvent)]):
        LABELS[solvent].append(name)
        POINTS_POSITION[solvent].append((etheo, eexp))
        LABELS_KWARGS[solvent].append(dict(color=color, ha='center', va='center'))

def plot_corr(ax, data: pandas.DataFrame, solvent: str):
    e = EXCLUDE.copy()
    
    if solvent == 'acetonitrile':
        e += [12, 4]
        
    x, y = data[~data['compound'].isin(e)]['E_ox_theo_{}'.format(solvent)], data[~data['compound'].isin(e)]['E_ox_exp_{}'.format(solvent)]
    result = scipy.stats.linregress(x, y)
    mae = numpy.mean(numpy.abs(x-y))
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = x.min()
    ax.text(x + .05, result.slope*x + result.intercept, '{:.2f} $\\times E^P_{{rel}}$ + {:.2f}\n($R^2$={:.2f}, MAE={:.2f} V)'.format(result.slope, result.intercept, result.rvalue **2, mae))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-i2', '--input2', default='../data/Data_pot_ox_exp.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_matsui.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)
data_exp = pandas.read_csv(args.input2)

figure = plt.figure(figsize=(5, 9))
ax1, ax2 = figure.subplots(2, 1)

subdata_wa, param_matsui_wa = prepare_data(data, data_exp, 'water')

ax1.text(.72, .95, '$E_{{SHE}}$ = {:.2f} V, $f$ = {:.3f}, $\\mu$ = {:.5f} a$_0^{{-1}}$'.format(*param_matsui_wa))

plot_exp_vs_matsui(ax1, subdata_wa, 'water', 'Family.P6O', 'tab:blue')
plot_exp_vs_matsui(ax1, subdata_wa, 'water', 'Family.P5O', 'black')

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

subdata_ac, param_matsui_ac = prepare_data(data, data_exp, 'acetonitrile')

ax2.text(.8, 1.18, '$E_{{SHE}}$ = {:.2f} V, $f$ = {:.3f}, $\\mu$ = {:.5f} a$_0^{{-1}}$'.format(*param_matsui_ac))

plot_exp_vs_matsui(ax2, subdata_ac, 'acetonitrile', 'Family.P6O', 'tab:blue')
plot_exp_vs_matsui(ax2, subdata_ac, 'acetonitrile', 'Family.P5O', 'black')
plot_exp_vs_matsui(ax2, subdata_ac, 'acetonitrile', 'Family.IIO', 'tab:green')
plot_exp_vs_matsui(ax2, subdata_ac, 'acetonitrile', 'Family.APO', 'tab:red')

plot_corr(ax2, subdata_ac, 'acetonitrile')

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

ax2.legend()
[ax.set_xlabel('Computed $E^P_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]
[ax.set_ylabel('Experimental $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]

plt.tight_layout()
figure.savefig(args.output)
