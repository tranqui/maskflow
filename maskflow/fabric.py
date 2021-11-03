#!/usr/bin/env python3

try: from .multifibre import *
except: from multifibre import *

from scipy.interpolate import interp1d

def mechanical_mobility(diameter, temperature=standard_temp):
    """
    Args:
        diameter: particle diameter (m)
        temperature: temperature of particle/environment (assumed contant) (K)
    Returns:
        Inertial parameter for particle in medium (s/kg).
    """
    return slip_correction(diameter, temperature) / (3*np.pi * medium.dynamic_viscosity(temperature) * diameter)

def diffusion_coefficient(diameter, temperature=standard_temp):
    """
    Args:
        diameter: particle diameter (m)
        temperature: temperature of particle/environment (assumed contant) (K)
    Returns:
        Diffusion coefficient of particle relative to fluid flow from Stokes-Einstein law (m^2/s).
    """
    return mechanical_mobility(diameter, temperature) * boltzmann_constant * temperature

class Fabric:
    def __init__(self,
                 inertia_data_path,
                 temperature = standard_temp,
                 leakage = 0):

        data = read_yaml(inertia_data_path)

        df = 1e-6*data['fibre_diameter']
        df_sigma = data['fibre_sigma']
        self.fibre_diameter_distribution = LogNormalFibreDistribution(np.log(df), df_sigma)

        self.volume_fraction = data['alpha']
        self.fibre_number_density = 4*self.volume_fraction/(np.pi*self.fibre_diameter_distribution.moment(2))

        self.flow_speed = data['flow_speed']
        self.face_speed = (1 - self.volume_fraction)*self.flow_speed
        self.mask_area = 190e-4 # m^2/s
        self.flow_rate = 60e3 * self.mask_area * self.face_speed / (1-leakage) # litres/min
        # print('flow_speed:', self.flow_speed)
        # print('face_speed:', self.face_speed)
        # print(' flow_rate:', self.flow_rate)

        self.inertial_lambda = LambdaInterpolator(1e-6*np.array(data['particle_diameters']),
                                                  1e-6*np.array(data['lambda']))

        self.temperature = temperature
        self.leakage = leakage

    def effective_single_fibre_lambda(self, particle_diameter,
                                      diffusion_pore_correction=1,
                                      include_diffusion=True,
                                      include_inertia=True):
        efficiency = include_diffusion*self.diffusion_lambda(particle_diameter, diffusion_pore_correction)
        if include_inertia: efficiency += self.inertial_lambda(particle_diameter)
        return efficiency

    def penetration_depth(self, particle_diameter,
                          diffusion_pore_correction=1,
                          include_diffusion=True, include_inertia=True):
        return (1 - self.volume_fraction) / (self.fibre_number_density * self.effective_single_fibre_lambda(particle_diameter, diffusion_pore_correction, include_diffusion, include_inertia))

    def penetration(self, particle_diameter, filter_thickness,
                    diffusion_pore_correction=1,
                    include_diffusion=True, include_inertia=True):
        return self.leakage + (1-self.leakage)*np.exp(-filter_thickness / self.penetration_depth(particle_diameter, diffusion_pore_correction, include_diffusion, include_inertia))

    def effectiveness(self, exhaled_droplet_distribution,
                      filter_thickness=1e-3,
                      exhale_covered=True, inhale_covered=True,
                      infectivity=lambda dp: 1,
                      evaporation_factor=3,
                      N=1000):
        P_func = lambda dp: self.penetration(dp, filter_thickness)

        inhaled_diameters = np.logspace(-8, -3, N)
        exhaled_diameters = evaporation_factor*inhaled_diameters

        unmitigated_inhaled_distribution = exhaled_droplet_distribution(1e6*evaporation_factor*exhaled_diameters)
        mitigated_inhaled_distribution = unmitigated_inhaled_distribution.copy()
        if exhale_covered:
            mitigated_inhaled_distribution *= P_func(exhaled_diameters)
        if inhale_covered:
            mitigated_inhaled_distribution *= P_func(inhaled_diameters)

        integrand = infectivity(inhaled_diameters)
        return integrate.simps(mitigated_inhaled_distribution * integrand) / integrate.simps(unmitigated_inhaled_distribution * integrand)

    @property
    def hydrodynamic_factor(self):
        return -0.5*np.log(self.volume_fraction) - 0.75 + self.volume_fraction - 0.25*self.volume_fraction**2

    # Single-fibre calculations.

    def df_power(self, n):
        """Helper function to determine the average value of the fibre diameter d_f."""
        return self.fibre_diameter_distribution.moment(n)

    def R_power(self, n):
        """Helper function to determine the average value of R^n (R=d_p/d_f)."""
        return self.particle_diameter**n * self.df_power(-n)

    # @property
    # def peclet_number(self):
    #     return self.fibre_diameter * self.flow_speed / diffusion_coefficient(self.particle_diameter, self.temperature)

    def diffusion_lambda(self, particle_diameter, diffusion_pore_correction=1):
        """Average the single-fibre result over the fibre distribution.

        We take the single-fibre result of Stechkina and Fuchs (1966):

             lambda/df = 2.9*(K*Pe**2)**(-1/3) + 0.624/Pe + 1.24*R**(2/3) / np.sqrt(K*Pe)

        where Pe is the particle Peclet number, R = d_p/d_f and K is the (Kuwabara) hydrodynamic
        factor.

        Args:
            particle_diameter: diameter of incoming particles (m)
            diffusion_pore_correction: amount to rescale flow speed/Peclet number by to account
                                       for the increased flow through inter-yarn pores
        Returns:
            lambda: characterising the effective efficiency of a single-fibre by the size of
                    its capture window (m)
        """
        K = self.hydrodynamic_factor
        Pe_div_df = diffusion_pore_correction * self.flow_speed / diffusion_coefficient(particle_diameter, self.temperature)
        lam = 2.9*(K*Pe_div_df**2)**(-1/3) * self.df_power(1-2/3) + 0.624/Pe_div_df + 1.24*particle_diameter**(2/3) / np.sqrt(K*Pe_div_df) * self.df_power(1-1/6)
        return lam

    # @property
    # def most_penetrating(self):
    #     select = np.argmin(self.single_efficiency)
    #     return self.particle_diameter[select], self.single_efficiency[select]

def gradient(f, x, dx=1e-8, args=[]):
    """Numerically take the gradient of a scalar function at point x.

    This uses a central finite difference scheme to numerically find the derivatives.
    The objective function must return a scalar, but the inputs can be either a scalar
    or a vector.

    Args:
        f: objective function to differentiate
        x: scalar or vector argument to evaluate the derivatives
        dx: precision of derivatives for the finite difference method
        args: additional arguments to f
    Returns:
        g: gradient of the function at point x, with the same dimensionality as input x
    """
    f0 = f(x, *args)
    try: g = np.empty((x.size,len(f0)), dtype=np.longdouble)
    except: g = np.empty(x.size, dtype=np.longdouble)
    shape = x.shape

    x = x.reshape(-1)
    q = g.copy()
    for i in range(x.size):
        x0 = x[i]
        x[i] += dx
        g[i] = (f(x.reshape(shape), *args)-f0) / (x[i]-x0) # using exact step after rounding
        q[i] = f(x.reshape(shape), *args)
        x[i] = x0

    try: return g.reshape(shape)
    except: return g

def penetration(paths, dp, flow_speed, df, df_sigma,
                thickness, alpha, pore_permeability=None,
                thickness_error=0, alpha_error=0, pore_permeability_error=0,
                return_error=False):
    """Calculate the penetration for a fabric of an arbitrary volume fraction by interpolating
    over available data.

    Args:
        paths: single-fibre efficiency data at various sampled volume fractions
        dp: particle diameter (m)
        flow_speed
        df
        df_sigma
        thickness (mm)
        alpha
        pore_permeability
        thickness_error
        alpha_error
        pore_permeability_error
        return_error: whether to calculate the error in the penetration.
    Returns
        the penetration at each particle diameter
        error in penetration (from propagating stated uncertainties) if return_error=True
    """
    if not return_error:
        sample_volume_fractions = [float(path.split('alpha=')[-1].split('_')[0]) for path in paths]

        p = []
        for path in paths:
            fabric = Fabric(path)
            if pore_permeability is not None:
                p += [fabric.penetration(dp, 1e-3*thickness,
                                         diffusion_pore_correction=1/pore_permeability)]
            else:
                p += [fabric.penetration(dp, 1e-3*thickness)]

        return interp1d(sample_volume_fractions, np.array(p).T,
                        'quadratic', fill_value='extrapolate')(alpha)

    else:
        f = lambda x: penetration(paths, dp, flow_speed, df, df_sigma, *x)

        if pore_permeability is not None:
            y = np.array((thickness, alpha, pore_permeability))
            y_err = np.array((1e-3*thickness_error, alpha_error, pore_permeability_error))
        else:
            y = np.array((thickness, alpha))
            y_err = np.array((1e-3*thickness_error, alpha_error))

        p = f(y)
        jacobian = gradient(f, y)
        cov_y = np.diag(y_err**2)
        p_err = np.sqrt(np.diag(jacobian.T.dot(cov_y.dot(jacobian))))

        return p, p_err

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dp2 = np.logspace(-8, -4, 1001)
    for path in sys.argv[1:]:
        fabric = Fabric(path)

        plt.semilogx(1e6*dp2, 1-fabric.penetration(dp2, 1e-3), label=(r'$\alpha = %.2f, U_0 = %.3f$' % (fabric.volume_fraction, fabric.flow_speed)))

    plt.legend(loc='best', fontsize=8)
    plt.xlabel('particle diameter $d_p$ (\si{\micro\meter})')
    plt.ylabel('collection efficiency')
    plt.xlim([1e-2, 1e2])
    plt.ylim([0, 1])

    plt.show()
