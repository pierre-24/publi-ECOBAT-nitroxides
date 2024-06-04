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
    return E_abs - 1/(2*f*a0 / AU_TO_ANG) * (1/epsilon_r-1)*((n_e-z)**2*scipy.special.erf(mu*a0*numpy.abs(n_e-z))-z**2*scipy.special.erf(mu*a0*numpy.abs(z))) * 27.212 - E_SHE # in V
    

def prepare_data(data: pandas.DataFrame, data_exp: pandas.DataFrame, solvent):
    subdata = data[data['solvent'] == solvent]
    subdata.insert(1, 'compound', [int(n.replace('mol_', '')) for n in subdata['name']])
    subdata = subdata.join(data_exp[data_exp['E_ox_exp_{}'.format(solvent)].notnull()].set_index('compound'), on='compound', how='inner')
    
    subdata['E_ox_exp_{}'.format(solvent)] /= 1000
    
    def matsui(x, E_SHE, f, mu):
        return matsui_E_P_rel(x[~subdata['compound'].isin(EXCLUDE)], E_SHE, f, mu, subdata[~subdata['compound'].isin(EXCLUDE)]['r_rad'], EPSILON_R[solvent], subdata[~subdata['compound'].isin(EXCLUDE)]['z'] + 1)
    
    (mat_E_SHE, mat_f, mat_mu), cov = scipy.optimize.curve_fit(
        matsui, 
        subdata[~subdata['compound'].isin(EXCLUDE)]['E_ox'], 
        subdata[~subdata['compound'].isin(EXCLUDE)]['E_ox_exp_{}'.format(solvent)],
        p0=[E_SHE_[solvent], 1.0, 0.1],
        method='dogbox'
    )
    
    subdata.insert(1, 'E_ox_theo_{}'.format(solvent), matsui_E_P_rel(subdata['E_ox'], mat_E_SHE, mat_f, mat_mu, subdata['r_rad'], EPSILON_R[solvent], subdata['z'] + 1))
    
    return subdata[['compound', 'family', 'E_ox_theo_{}'.format(solvent), 'E_ox_exp_{}'.format(solvent)]], (mat_E_SHE, mat_f, numpy.abs(mat_mu))
    

def plot_exp_vs_matsui(ax, data: pandas.DataFrame, solvent: str, family: str, color: str):
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
    
    x = numpy.array( [x.min(), x.max()])
    ax.plot(x, result.slope*x + result.intercept, 'k--')
    
    x = .9 * x.min()+ .1 * x.max()
    ax.text(x + .05, result.slope*x + result.intercept, '{:.2f} $\\times E^0_{{rel}}$ + {:.2f} ($R^2$={:.2f})'.format(result.slope, result.intercept,result.rvalue **2))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../data/Data_pot.csv')
parser.add_argument('-i2', '--input2', default='../data/Data_pot_ox_exp.csv')
parser.add_argument('-r', '--reposition-labels', action='store_true')
parser.add_argument('-o', '--output', default='Data_pot_matsui.pdf')

args = parser.parse_args()

data = pandas.read_csv(args.input)
data_exp = pandas.read_csv(args.input2)

figure = plt.figure(figsize=(10, 5))
ax1, ax2 = figure.subplots(1, 2)

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

ax2.text(.8, 1.18, '$E_{{SHE}}$ = {:.2f} V, $f$ = {:.1f}, $\\mu$ = {:.2f} a$_0^{{-1}}$'.format(*param_matsui_ac))

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
    positioner.optimize(dx=1e-3, beta=1e4, krep=1, kspring=1000, c=0.05, b0=0.015)
    positioner.save(LABELS_PATH['acetonitrile'])

positioner.add_labels(ax2)

ax2.legend()
[ax.set_xlabel('Computed $E^P_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]
[ax.set_ylabel('Experimental $E^0_{rel}(N^+|N^\\bullet)$ (V)') for ax in (ax1, ax2)]

plt.tight_layout()
figure.savefig(args.output)