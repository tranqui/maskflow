#!/usr/bin/env python3

import sys, os
import numpy as np
from scipy import optimize

try:
    from .kuwabara import KuwabaraFlowField, penetration
    from . import air as medium
except:
    from kuwabara import KuwabaraFlowField, penetration
    import air as medium

water_density = 998 # kg/m^3, at room temperature
particle_density = water_density

elementary_charge = 1.602e-19 # C
boltzmann_constant = 1.38e-23 # J / K
vacuum_permittivity = 8.854e-12 # F / m

water_freezing = 273.15 # K
standard_temp = water_freezing + 20 # room temperature
body_temp = water_freezing + 37

def slip_correction(diameter, temperature, A1=2.492, A2=0.84, A3=0.435):
    """
    Slip correction for Stokes flow past a sphere (of Cunningham form).

    We take default parameters from Kanaoka (1987) DOI: 10.1080/02786828708959142.

    Args:
        diameter: particle diameter (m)
        A1, A2, A3: empirical correction parameters.
    Returns:
        Correction for slip effects in Stokes flow past a sphere.
    """
    l = medium.mean_free_path(temperature)
    return 1 + l/diameter * (A1 + A2 * np.exp(-A3*diameter/l))

def standard_stokes_number(particle_diameter, fibre_diameter,
                           flow_speed, temperature=standard_temp):
    """
    Args:
        particle_diameter: particle diameter (m)
        fibre_diameter: fibre diameter (m)
        flow_speed: speed of flow (far field) relative to object (m/s)
        temperature: temperature of particle/environment (assumed contant) (K)
    Returns:
        Stokes number for particle transport through medium.
    """
    return particle_density * particle_diameter**2 * flow_speed * slip_correction(particle_diameter, temperature) / (9 * medium.dynamic_viscosity(temperature) * fibre_diameter)

def refine_theta(flow, R, St, th1, th2, tmax, max_step, verbose=True):
    th = 0.5*(th1 + th2)
    collides = flow.does_collide(R, St, th, tmax=tmax, max_step=max_step)
    if collides: th2 = th
    else: th1 = th
    if verbose: sys.stderr.write('updated bounds: %.8g %.8g %r\n' % (th1, th2, collides))
    return th1, th2

def find_theta(niters, flow, R, St, tmax, max_step, verbose=True, return_angles=False):
    if verbose: sys.stderr.write('beginning iterations for R=%.8g St=%.8g...\n' % (R, St))

    theta1 = np.pi/2
    theta2 = np.pi
    for _ in range(niters):
        theta1, theta2 = refine_theta(flow, R, St, theta1, theta2, tmax, max_step, verbose)

    lam1 = flow.lambda_from_theta0(theta1)
    lam2 = flow.lambda_from_theta0(theta2)

    if return_angles: return theta1, theta2, lam1, lam2
    else: return lam1, lam2

def find_lambda(niters, flow, R, stokes, tmax, max_step, verbose=True):
    rmax = 1 / np.sqrt(flow.alpha) - 1
    lmax = 2 / np.sqrt(flow.alpha)
    f = np.vectorize(lambda r,st: find_theta(niters, flow, r, st, tmax, max_step, verbose) if r < rmax else (lmax, lmax), signature='(),()->(),()')
    #f = np.vectorize(lambda r,st: find_theta(niters, flow, r, st, tmax, max_step, verbose) if r < rmax else (flow.interception_lambda(r), flow.interception_lambda(r)), signature='(),()->(),()')
    lam1, lam2 = f(R, stokes)

    lam = 0.5 * (lam1 + lam2)
    lam_error = 0.5 * np.abs(lam2 - lam1)

    return lam, lam_error

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="""Determine single-fibre efficiency by finding the limiting trajectory. This involves optimising the initial condition to find the trajectory that just glances the fibre.

    By default efficiency is stated in terms of a parameter 'lambda', which measures the width of streamlines that result in collection at the fibre surface.""")

    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='print status messages to stderr detailing iteration steps')
    parser.add_argument('-n', '--niters', type=int, default=25,
                        help='number of iterations (trajectories) in optimisation procedure which determines the error in the method (each iteration adds approximately one bit of accuracy) (default=25)')
    parser.add_argument('-t', '--time', type=float, default=1e3,
                        help='maximum integration time in simulation units (default=1e3)')
    parser.add_argument('-s', '--step', type=float, default=0.1,
                        help='maximum step size of simulation (default=0.1)')
    parser.add_argument('-f', '--fibre-radius', dest='fibre_radius',
                        type=float, default=1, help='radius of fibre (default=1)')
    parser.add_argument('-d', '--diameter', action='store_true',
                        help='stated length parameters are diameters rather than radii')
    parser.add_argument('-R', '--rescale', action='store_true',
                        help='rescale efficiency in terms of the fibre diameter')
    parser.add_argument('-S', '--stokes', type=float,
                        help='override the Stokes number for particle inertia (otherwise it is calculated from other parameters, assuming sizes are stated in microns)')
    parser.add_argument('-F', '--flow-speed', type=float, default=0.027,
                        help='flow speed (m/s) used in Stokes number calculation (default=0.027)')
    parser.add_argument('-P', '--penetration', type=float, default=-1,
                        help='final penetration through filter of given thickness')
    parser.add_argument('-Y', '--yaml', action='store_true',
                        help='output in a fully YAML-compliant format')

    parser.add_argument('-q', '--quick', action='store_false', dest='full', default=True,
                        help='skip full integration method which is generally slower than the other approximate methods')
    parser.add_argument('-e', '--error', action='store_true',
                        help='display error in estimated efficiency in standard method')
    parser.add_argument('-i', '--interception', action='store_true',
                        help='show efficiency in pure interception mechanism')
    parser.add_argument('-a', '--analytical', action='store_true',
                        help='evaluate the efficiency with approximate analytical formulas rather than by numerical integration')
    parser.add_argument('-p', '--perturbative', action='store_true',
                        help='evaluate the efficiency using approximate perturbative theory')

    parser.add_argument('radius', help='radius of particle (relative to fibre unless specified with -f)')
    parser.add_argument('alpha', type=float, help='volume fraction of filter')

    args = parser.parse_args()

    if args.yaml:
        # If enabled force YAML compliant output by removing whitespace at start of printed data.
        import builtins as __builtin__
        def print(*args, **kwargs):
            __builtin__.print(*[a.strip() if type(a) == str else a for a in args])

    args.radius = eval(args.radius)
    if args.diameter:
        args.radius = args.radius / 2
        args.fibre_radius = args.fibre_radius / 2
    R = args.radius / args.fibre_radius

    if not args.stokes:
        args.stokes = standard_stokes_number(2e-6*args.radius, 2e-6*args.fibre_radius, args.flow_speed)

    flow = KuwabaraFlowField(args.alpha)

    np.set_printoptions(12, suppress=True, linewidth=np.nan)
    # add a comma between entries to aid parsing
    np.set_string_function(lambda x: repr(x).replace('(', '').replace(')', '').replace('array', '').replace("       ", ' ') , repr=False)

    print('             particle_radius:', args.radius)
    print('                fibre_radius:', args.fibre_radius)
    print('                       alpha:', args.alpha) 
    print('         hydrodynamic_factor:', flow.hydrodynamic_factor)
    print('                  flow_speed:', args.flow_speed)
    print('                      stokes:', args.stokes)
    print('              outer_boundary:', flow.l * args.fibre_radius)
    print()

    if args.full:
        lam, lam_error = find_lambda(args.niters, flow, R, args.stokes, args.time, args.step, args.verbose)

        lam *= args.fibre_radius
        lam_error *= args.fibre_radius

        print('                      lambda:', lam)
        if args.error: print('                       error:', lam_error)

    if args.interception:
        interception_lam = args.fibre_radius * flow.interception_lambda(R)
        print('         interception_lambda:', interception_lam)

    if args.analytical:
        stechkina_lam = args.fibre_radius * flow.stechkina_lambda(R, st)
        print('            stechkina_lambda:', stechkina_lam)

    if args.perturbative:
        f = np.vectorize(lambda r,st: flow.perturbative_impaction_efficiency(r, st), signature='(),()->()')
        perturb_lam = args.fibre_radius * f(R, args.stokes)
        print('         perturbative_lambda:', perturb_lam)

    if args.rescale:
        print()
        if args.full:
            lam_rescaled = lam / (2*args.fibre_radius)
            lam_rescaled_error = lam / (2*args.fibre_radius)
            print('             rescaled_lambda:', lam / (2*args.fibre_radius))
            if args.error: print('         rescaled_lambda_err:', lam_error / (2*args.fibre_radius))
        if args.interception: print('rescaled_interception_lambda:', interception_lam / (2*args.fibre_radius))
        if args.analytical: print('   rescaled_stechkina_lambda:', stechkina_lam / (2*args.fibre_radius))
        if args.perturbative: print('rescaled_perturbative_lambda:', perturb_lam / (2*args.fibre_radius))

    if args.penetration > 0:
        print()
        print('              mask_thickness:', args.penetration)
        if args.full: print('                 penetration:', penetration(lam, args.penetration, 2*args.fibre_radius, args.alpha))
        if args.interception: print('    interception_penetration:', penetration(interception_lam, args.penetration, 2*args.fibre_radius, args.alpha))
        if args.analytical: print('       stechkina_penetration:', penetration(stechkina_lam, args.penetration, 2*args.fibre_radius, args.alpha))
        if args.perturbative: print('         perturb_penetration:', penetration(perturb_lam, args.penetration, 2*args.fibre_radius, args.alpha))
