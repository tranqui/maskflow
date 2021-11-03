#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
import pandas
from scipy import integrate
from scipy.interpolate import interp1d

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
deposition_data = '%s/chalvatzaki2020.xls' % dir_path

suffixes = ['extrathoracic', 'tracheobronchial', 'alveolar-interstitial']

deposition_functions = {}
mouthpiece_diameter = 0.79
for sheet in [2, 0, 1]:
    suffix = suffixes[sheet]

    table = pandas.read_excel(deposition_data, index_col=0, sheet_name=sheet)
    x = table.columns.values
    y = 1e-2 * table.loc['%.2f cm mouthpiece diameter' % mouthpiece_diameter].values

    deposition_functions[suffix] = interp1d(x, y, fill_value='extrapolate', kind='quadratic')

def deposition_probability(D):
    try: p = np.zeros(D.shape)
    except: p = 0
    for _,f in deposition_functions.items(): p += f(D)
    return p

if __name__ == '__main__':
    from blomodel import *
    modes = {r'normal speech': Gregson2020.Speaking70to80dB,
             'voluntary cough': Johnson2011.Coughing}

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    D = np.geomspace(0.1, 10, 1001)
    ax1.plot(D, deposition_probability(D), '--', label='total')

    for suffix, f in deposition_functions.items():
        pl, = ax1.plot(D, f(D), '-' , label=suffix)
        ax1.plot(f.x, f(f.x), 'o', mfc='None', c=pl.get_color())

    for mode_name, mode in modes.items():
        p = mode.measured_aerosol_distribution(D)
        pl, = ax2.plot(D, p, '-', label=mode_name)
        p *= deposition_probability(D)
        p /= integrate.simps(p, np.log(D))
        ax2.plot(D, p, '--', c=pl.get_color())

    ax2.annotate('all exhaled\naerosols',
                 [0.45, 0.68], [0.2, 0.9],
                 fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5),
                 horizontalalignment='center', verticalalignment='center')

    ax2.annotate('deposited\naerosols',
                 [1.76, 0.48], [3.5, 0.7],
                 fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5),
                 horizontalalignment='center', verticalalignment='center')

    ax1.set_ylabel(r'deposition probability')
    ax2.set_ylabel('pdf')

    ax2.set_xlabel(r'particle diameter $d_p$ (\si{\micro\metre})')
    ax2.set_ylim([0, 1.5])

    ax1.legend(loc='best')
    ax2.legend(loc='best')

    ax1.set_xscale('log')
    ax1.set_xlim([0.1, 10])
    ax1.set_ylim([0, 1])

    plt.show()
