#!/usr/bin/env python3

try: from .singlefibre import *
except: from singlefibre import *

from scipy import integrate
from scipy import interpolate

import yaml
def read_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.Loader)

class LambdaInterpolator:
    def __init__(self, d, lam, extend_orders=5, eps=1e-6, extend_N=2):
        self.d = d.copy()
        self.lam = lam.copy()
        logd = np.log(self.d)

        logd1, logd2 = logd[:2]
        lam1, lam2 = lam[:2]
        exponent0 = np.log(lam2/lam1) / (logd2 - logd1)
        logd_before = logd1 + np.log(np.logspace(-extend_orders, 0, extend_N) - eps)
        lam_before = np.exp( (logd_before - logd1) * exponent0 ) * lam1

        logd1, logd2 = logd[-2:]
        lam1, lam2 = lam[-2:]
        exponent1 = np.log(lam2/lam1) / (logd2 - logd1)
        logd_after = logd2 + np.log(np.logspace(0, extend_orders, extend_N) + eps)
        lam_after = np.exp( (logd_after - logd2) * exponent1 ) * lam2

        logd_range = np.concatenate((logd_before, logd, logd_after))
        lam_range = np.concatenate((lam_before, lam, lam_after))

        self.loglam_func = interpolate.interp1d(logd_range, np.log(lam_range),
                                                'quadratic', fill_value='extrapolate')
        self.lam_func = lambda d: np.exp(self.loglam_func(np.log(d)))

    def __call__(self, d):
        return self.lam_func(d)


class FibreDistribution:
    def moment(self, n):
        raise NotImplementedError

    def __call__(self, d):
        return self.pdf(d)

    @property
    def pdf(self):
        raise NotImplementedError

    def average(self, f):
        p = self.pdf
        return integrate.quad(lambda d: p(d)*f(d), 0, 5*self.moment(1))[0]

class LogNormalFibreDistribution(FibreDistribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def moment(self, n):
        return np.exp(n*self.mu + 0.5*(n*self.sigma)**2)

    @property
    def pdf(self):
        return lambda d: np.exp(-0.5*(np.log(d)-self.mu)**2/self.sigma**2) / (d*self.sigma*np.sqrt(2*np.pi))

    @property
    def log_pdf(self):
        return lambda x: np.exp(-0.5*(x-self.mu)**2/self.sigma**2) / (self.sigma*np.sqrt(2*np.pi))

    def average(self, f, n=30):
        if self.sigma == 0: return f(np.exp(self.mu))
        p = self.log_pdf
        return integrate.fixed_quad(lambda d: p(d)*f(np.exp(d)), self.mu-3, self.mu+3, n=n)[0]

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="""Determine efficiency of fabric by integrating over distribution of single fibres.""")

    parser.add_argument('-p', '--plot', action='store_true',
                        help='plot lambda values obtained from integration')
    parser.add_argument('-P', '--penetration', type=float, default=-1,
                        help='plot final penetration through filter of given thickness')

    parser.add_argument('fibre_diameter', metavar='df', type=float,
                        help='modal value of fibre diameter (in microns)')
    parser.add_argument('fibre_sigma', metavar='df_sigma', type=float,
                        help='standard deviation of ln(df/micron)')

    parser.add_argument('data_files', nargs='+',
                        help='YAML files containing data on single-fibre capture efficiencies')

    parser.add_argument('-Y', '--yaml', action='store_true',
                        help='output in a fully YAML-compliant format')

    args = parser.parse_args()

    df_distribution = LogNormalFibreDistribution(np.log(args.fibre_diameter), args.fibre_sigma)

    #from natsort import natsorted
    data = [read_yaml(path) for path in args.data_files]
    data = [d for d in data if 'lambda' in d]
    data = sorted(data, key=lambda d: d['fibre_radius'])

    alpha = data[0]['alpha']
    flow_speed = data[0]['flow_speed']
    K = data[0]['hydrodynamic_factor']
    particle_diameters = 2*np.array(data[0]['particle_radius'])
    sampled_fibre_diameters = 2*np.array([d['fibre_radius'] for d in data])
    sampled_lambdas = np.array([np.array(d['lambda']) for d in data])

    for df,lam in zip(sampled_fibre_diameters, sampled_lambdas):
        Rmax = (1 / np.sqrt(alpha) - 1)
        dpmax = df * Rmax
        #lam_max = df / np.sqrt(alpha)
        select = particle_diameters < dpmax

        # Extrapolate beyond Kuwabara regime with a simple linear fit:
        # We do not require accuracy in this region as the penetration is so small, however we
        # do require it to decrease monotonically to zero and a linear fit suffices for this.
        dp1, dp2 = particle_diameters[select][-2:]
        lam1, lam2 = lam[select][-2:]
        dp_after = particle_diameters[~select]
        lam[~select] = lam2 + (lam2-lam1) * (dp_after-dp2)/(dp2-dp1) #l[select][-1]

    averaged_lambda = np.empty(particle_diameters.shape)

    for i,dp in enumerate(particle_diameters):
        lam = sampled_lambdas[:,i]
        logdf = np.log(sampled_fibre_diameters)
        lam_func = LambdaInterpolator(sampled_fibre_diameters, sampled_lambdas[:,i])
        averaged_lambda[i] = df_distribution.average(lam_func) #lambda x: lam_func(1e6*x))

    if args.plot:
        plt.loglog(particle_diameters, averaged_lambda)
        plt.xlabel('particle diameter $d_p$ (\si{\micro\meter})')
        plt.ylabel(r'$\langle \lambda \rangle$')

    fibre_number_density = 4*alpha/(np.pi*df_distribution.moment(2))
    penetration_length = (1 - alpha) / (fibre_number_density * averaged_lambda)

    if args.penetration > 0:
        plt.figure()
        filter_thickness = args.penetration
        P = np.exp(-filter_thickness / penetration_length)
        plt.semilogx(particle_diameters, 1-P)
        plt.xlabel('particle diameter $d_p$ (\si{\micro\meter})')
        plt.ylabel('$1 - P$')
        plt.title('$L = \SI{1}{\milli\meter}$')

    if args.yaml:
        # If enabled force YAML compliant output by removing whitespace at start of printed data.
        import builtins as __builtin__
        def print(*args, **kwargs):
            __builtin__.print(*[a.strip() if type(a) == str else a for a in args])

    np.set_printoptions(12, suppress=True, linewidth=np.nan)
    # add a comma between entries to aid parsing
    np.set_string_function(lambda x: repr(x).replace('(', '').replace(')', '').replace('array', '').replace("       ", ' ') , repr=False)

    print('      fibre_diameter:', args.fibre_diameter)
    print('         fibre_sigma:', args.fibre_sigma)
    print('               alpha:', alpha)
    print('          flow_speed:', flow_speed)
    print('  particle_diameters:', particle_diameters)
    print('              lambda:', averaged_lambda)
    print('  penetration_length:', penetration_length)

    plt.show()
