#!/usr/bin/env python3

import sys, os
import numpy as np
from scipy import optimize
#from impact_kuwabara import does_collide, efficiency
#from impaction import does_collide, efficiency_from_theta, perturbative_impaction_efficiency
from kuwabara import KuwabaraFlowField, penetration

import air as medium

water_density = 997 # kg/m^3
particle_density = water_density

elementary_charge = 1.602e-19 # C
boltzmann_constant = 1.38e-23 # J / K
vacuum_permittivity = 8.854e-12 # F / m

water_freezing = 273.15 # K
standard_temp = water_freezing + 20
body_temp = water_freezing + 37

def slip_correction(diameter, temperature, A1=2.492, A2=0.84, A3=0.435):
    """
    Slip correction for Stokes flow past a sphere (of Cunningham form).

    Args:
        diameter: particle diameter (m)
        A1, A2, A3: empirical correction parameters.
    Returns:
        Correction for slip effects in Stokes flow past a sphere.
    """
    l = medium.mean_free_path(temperature)
    return 1 + l/diameter * (A1 + A2 * np.exp(-A3*diameter/l))

def standard_stokes_number(particle_diameter, fibre_diameter,
                           flow_speed=0.1, temperature=standard_temp):
    """
    Args:
        particle_diameter: particle diameter (m)
        fibre_diameter: fibre diameter (m)
        flow_speed: speed of flow (far field) relative to object (m/s)
        temperature: temperature of particle/environment (assumed contant) (K)
    Returns:
        Stokes number for particle transport through medium.
    """
    return particle_density * particle_diameter**2 * flow_speed * slip_correction(particle_diameter, temperature) / (18 * medium.dynamic_viscosity(temperature) * fibre_diameter)

def refine_theta(flow, R, St, th1, th2, tmax, max_step, verbose=True):
    th = 0.5*(th1 + th2)
    collides = flow.does_collide(R, St, th, tmax=tmax, max_step=max_step)
    if collides: th2 = th
    else: th1 = th
    if verbose: print('updated bounds: %.8g %.8g' % (th1, th2), collides)
    return th1, th2

def find_theta(niters, flow, R, St, tmax, max_step, verbose=True):
    if verbose: print('beginning iterations for St=%.8g...' % St)

    theta1 = np.pi/2
    theta2 = np.pi
    for _ in range(niters):
        theta1, theta2 = refine_theta(flow, R, St, theta1, theta2, tmax, max_step, verbose)

    lam1 = flow.lambda_from_theta0(theta1)
    lam2 = flow.lambda_from_theta0(theta2)

    return theta1, theta2, lam1, lam2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""Determine single-fibre efficiency by finding the limiting trajectory. This involves optimising the initial condition to find the trajectory that just glances the fibre.

    By default efficiency is stated in terms of a parameter 'lambda', which measures the width of streamlines that result in collection at the fibre surface.""")

    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='print status messages to stdout detailing iteration steps')
    parser.add_argument('-n', '--niters', type=int, default=25,
                        help='number of iterations (trajectories) in optimisation procedure which determines the error in the method (each iteration adds approximately one bit of accuracy)')
    parser.add_argument('-t', '--time', type=float, default=1e3,
                        help='maximum integration time (in simulation units)')
    parser.add_argument('-s', '--step', type=float, default=0.1,
                        help='maximum step size of simulation')
    parser.add_argument('-f', '--fibre-radius', dest='fibre_radius',
                        type=float, default=1, help='radius of fibre')
    parser.add_argument('-d', '--diameter', action='store_true',
                        help='stated length parameters are diameters rather than radii')
    parser.add_argument('-S', '--stokes', type=float,
                        help='override the Stokes number for particle inertia (otherwise it is calculated from other parameters, assuming sizes are stated in microns)')
    parser.add_argument('-P', '--penetration', type=float, default=-1,
                        help='final penetration through filter of given thickness')
    
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument('-e', '--error', action='store_true',
                              help='display error in estimated efficiency in standard method')
    method_group.add_argument('-a', '--analytical', action='store_true',
                              help='evaluate the efficiency with approximate analytical formulas rather than by numerical integration')
    method_group.add_argument('-p', '--perturbative', action='store_true',
                              help='evaluate the efficiency using approximate perturbative theory')

    parser.add_argument('radius', type=float, help='radius of particle (relative to fibre unless specified with -f)')
    parser.add_argument('alpha', type=float, help='volume fraction of filter')

    args = parser.parse_args()
    if args.diameter:
        args.radius /= 2
        args.fibre_radius /= 2
    R = args.radius / args.fibre_radius

    if not args.stokes:
        args.stokes = standard_stokes_number(2e-6*args.radius, 2e-6*args.fibre_radius)

    flow = KuwabaraFlowField(args.alpha)

    if args.analytical:
        lam1 = flow.stechkina_lambda(R, args.stokes)
        lam2 = lam1
        theta1 = theta2 = 0
    elif args.perturbative:
        theta1, _, lam1 =  flow.perturbative_impaction_efficiency(R, args.stokes, return_angles=True)
        lam2 = lam1
        theta2 = theta1
    else:
        theta1, theta2, lam1, lam2 = find_theta(args.niters, flow, R, args.stokes, args.time, args.step, args.verbose)

    lam1 *= args.radius
    lam2 *= args.radius
    lam = 0.5 * (lam1 + lam2)
    lam_error = 0.5 * np.abs(lam2 - lam1)

    np.set_printoptions(12)
    print('     lambda:', lam)
    if args.error: print('      error:', lam_error)
    if args.penetration > 0: print('penetration:', penetration(lam, args.penetration, 2*args.radius, args.alpha))

