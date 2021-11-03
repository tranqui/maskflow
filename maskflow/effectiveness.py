#!/usr/bin/env python3
"""This script calculates practical mask effectiveness for a surgical mask.
We calculate this as the (number) fraction of aerosols removed that:
  (i) contain at least one virion, which depends on the viral load of the exhaler, and
  (ii) actually deposit in the respiratory tract of the inhaler.
"""

import numpy as np, matplotlib.pyplot as plt
from surgicalmask import SurgicalMask4
from scipy import interpolate

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder = '%s/surgicalmaskdata' % dir_path
flow_speed = 0.063

D = np.logspace(-7, -5, 1001)
P = SurgicalMask4.penetration(D, flow_speed, data_folder)
mask_penetration = interpolate.interp1d(D, P, kind='quadratic', bounds_error=False, fill_value=0)
#mask_penetration = lambda d: SurgicalMask4.penetration(d, flow_speed, data_folder)
penetration = lambda d,a: a + (1-a) * mask_penetration(1e-6*d)

from deposition import deposition_probability

V = lambda d: np.pi*d**3 / 6 # micron^3
V_ml = lambda d: 1e-9 * V(d) # ml
mu = lambda d,phi: V_ml(d) * phi
viral_frac = lambda d,phi: 1 - np.exp(-mu(d,phi))
evap_factor = 3

def effectiveness(p, v, inhale_leakage=0, exhale_leakage=0):
    kernel = lambda d: deposition_probability(d) * viral_frac(d, v)
    weight = p.average(kernel)

    inhale_penetration = lambda d: penetration(d, inhale_leakage)
    exhale_penetration = lambda d: penetration(evap_factor*d, exhale_leakage)

    R = 1 - p.average(lambda d: kernel(d) * inhale_penetration(d) * exhale_penetration(d)) / weight
    return R

viral_loads = np.logspace(4, 16, 101)

fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
ax1, ax2, ax3 = axes

try: from .blomodel import Gregson2020, Johnson2011
except: from blomodel import Gregson2020, Johnson2011
modes = {r'normal speech': Gregson2020.Speaking70to80dB,
         r'voluntary cough': Johnson2011.Coughing}

first = True
for name, mode in modes.items():
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0, 1))
    pl, = ax1.semilogx(viral_loads, f(viral_loads), label=name)
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0.2, 1))
    ax1.semilogx(viral_loads, f(viral_loads), '--', c=pl.get_color())
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0.4, 1))
    ax1.semilogx(viral_loads, f(viral_loads), ':', c=pl.get_color())

    label = None
    if first: label = 'no leakage'
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 1, 0))
    pl, = ax2.semilogx(viral_loads, f(viral_loads), label=label)

    if first: label = '20\% leakage'
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 1, 0.2))
    ax2.semilogx(viral_loads, f(viral_loads), '--', c=pl.get_color(), label=label)

    if first: label = '40\% leakage'
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 1, 0.4))
    ax2.semilogx(viral_loads, f(viral_loads), ':', c=pl.get_color(), label=label)

    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0, 0))
    pl, = ax3.semilogx(viral_loads, f(viral_loads))
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0.2, 0.2))
    ax3.semilogx(viral_loads, f(viral_loads), '--', c=pl.get_color())
    f = np.vectorize(lambda v: effectiveness(mode.measured_aerosol_distribution, v, 0.4, 0.4))
    ax3.semilogx(viral_loads, f(viral_loads), ':', c=pl.get_color())

    first = False

ax1.legend(loc='lower right')
ax2.legend(loc='lower left')
ax3.set_xlabel('viral load $v$ (copies/\si{\milli\litre})')
for ax in axes: ax.set_ylabel('$R_\mathrm{vec}$')
ax1.set_xlim([viral_loads[0], viral_loads[-1]])
ax1.set_ylim([0.3, 1.0])

ax1.text(0.1, 0.055, 'masked inhaler',
         transform=ax1.transAxes, fontsize=10,
         horizontalalignment='left', verticalalignment='bottom')
ax2.text(0.9, 0.055, 'masked exhaler',
         transform=ax2.transAxes, fontsize=10,
         horizontalalignment='right', verticalalignment='bottom')
ax3.text(0.1, 0.055, 'both masked',
         transform=ax3.transAxes, fontsize=10,
         horizontalalignment='left', verticalalignment='bottom')

plt.show()
