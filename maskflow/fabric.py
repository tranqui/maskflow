#!/usr/bin/env python3

try: from .multifibre import *
except: from multifibre import *

from charge import thomson

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
                 leakage = 0,
                 surface_charge_density = 0,
                 eps_particle = 80,
                 eps_fibre = 2.5):

        data = read_yaml(inertia_data_path)

        df = 1e-6*data['fibre_diameter']
        df_sigma = data['fibre_sigma'] * 0
        self.fibre_diameter_distribution = LogNormalFibreDistribution(np.log(df), df_sigma)

        self.volume_fraction = data['alpha']
        self.fibre_number_density = 4*self.volume_fraction/(np.pi*self.fibre_diameter_distribution.moment(2))

        self.flow_speed = data['flow_speed']
        self.face_speed = (1 - self.volume_fraction)*self.flow_speed
        self.mask_area = 190e-4 # m^2/s
        self.flow_rate = 60e3 * self.mask_area * self.face_speed / (1-leakage) # litres/min
        print('flow_speed:', self.flow_speed)
        print('face_speed:', self.face_speed)
        print(' flow_rate:', self.flow_rate)

        self.inertial_lambda = LambdaInterpolator(1e-6*np.array(data['particle_diameters']),
                                                  1e-6*np.array(data['lambda']))

        self.temperature = temperature
        self.leakage = leakage

        self.surface_charge_density = surface_charge_density
        self.eps_particle = eps_particle
        self.eps_fibre = eps_fibre

    def electrostatic_lambda(self, particle_diameter):
        if self.surface_charge_density == 0: return 0

        n = np.arange(1001)
        prefactor = elementary_charge**2 / (4*np.pi*vacuum_permittivity * self.eps_particle * particle_diameter * boltzmann_constant * self.temperature)

        boltzmann_weight = np.empty((n.size, particle_diameter.size))
        for i in range(1,n.size):
            boltzmann_weight[i] = 2*np.exp(-prefactor * thomson(n[i]))
        boltzmann_weight[0] = 1
        boltzmann_weight /= np.sum(boltzmann_weight, axis=0)
        #typical_particle_charge = np.array([np.sum(B * n) for B in boltzmann_weight.T])
        #typical_particle_charge = n.dot(boltzmann_weight)

        flow = KuwabaraFlowField(self.volume_fraction)
        interception_lambda = lambda R: flow.interception_lambda(R)

        B = lambda dp: mechanical_mobility(dp, self.temperature)
        xF = lambda dp,df,m: B(dp)*abs(m)*elementary_charge*abs(self.surface_charge_density)*df / (vacuum_permittivity * (1 + self.eps_fibre) * self.flow_speed * (1 + dp/df) * interception_lambda(dp/df))

        coulombic_lambda1 = lambda x: (1 + x + x**2) / (1 + x) - 1
        coulombic_lambda2 = lambda dp,df,m: coulombic_lambda1(xF(dp,df,m))

        electrostatic_lambda = np.empty((n.size, particle_diameter.size))
        for m in n:
            coulomb_func = np.vectorize(lambda dp: self.fibre_diameter_distribution.average(lambda df: coulombic_lambda2(dp,df,m)))
            electrostatic_lambda[m] = boltzmann_weight[m] * coulomb_func(particle_diameter)
            #print(np.array((particle_diameter, 1e6*coulombic_lambda(particle_diameter))).T)
        electrostatic_lambda = np.sum(electrostatic_lambda, axis=0)
        #print(np.array((particle_diameter, coulombic_lambda)).T)

        #import sys; sys.exit(0)
        #coulombic_lambda = np.vectorize(lambda dp
        #coulombic_lambda = lambda m,R: Kc*m * (1/(1+R) - interception(R) / ((1+R)*interception(R) + Kc*m))

        # prefactor = ( (1 - self.volume_fraction)/self.Ku )**(1/8)
        # efficiency = prefactor * (np.pi*Kc) / (1 + 2*np.pi*Kc**0.25)

        return electrostatic_lambda

    def effective_single_fibre_lambda(self, particle_diameter):
        return self.diffusion_lambda(particle_diameter) + self.inertial_lambda(particle_diameter) + self.electrostatic_lambda(particle_diameter)

    def penetration_depth(self, particle_diameter):
        return (1 - self.volume_fraction) / (self.fibre_number_density * self.effective_single_fibre_lambda(particle_diameter))

    def penetration(self, particle_diameter, filter_thickness):
        return self.leakage + (1-self.leakage)*np.exp(-filter_thickness / self.penetration_depth(particle_diameter))

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

    def diffusion_lambda(self, particle_diameter):
        """Average the single-fibre result over the fibre distribution.

        We take the single-fibre result of Stechkina and Fuchs (1966):

             lambda/df = 2.9*(K*Pe**2)**(-1/3) + 0.624/Pe + 1.24*R**(2/3) / np.sqrt(K*Pe)

        where Pe is the particle Peclet number, R = d_p/d_f and K is the (Kuwabara) hydrodynamic
        factor.

        Args:
            particle_diameter: diameter of incoming particles (m)
        Returns:
            lambda: characterising the effective efficiency of a single-fibre by the size of
                    its capture window (m)
        """
        K = self.hydrodynamic_factor
        Pe_div_df = self.flow_speed / diffusion_coefficient(particle_diameter, self.temperature)
        lam = 2.9*(K*Pe_div_df**2)**(-1/3) * self.df_power(1-2/3) + 0.624/Pe_div_df + 1.24*particle_diameter**(2/3) / np.sqrt(K*Pe_div_df) * self.df_power(1-7/6)
        return lam

    # @property
    # def most_penetrating(self):
    #     select = np.argmin(self.single_efficiency)
    #     return self.particle_diameter[select], self.single_efficiency[select]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dp2 = np.logspace(-8, -4, 101)
    for path in sys.argv[1:]:
        fabric = Fabric(path)
        #fabric.effective_single_fibre_lambda(np.logspace(-7, -5, 11))

        pl, = plt.semilogx(1e6*dp2, 1-fabric.penetration(dp2, 1e-3), label=(r'$\alpha = %.2f, U_0 = %.3f$' % (fabric.volume_fraction, fabric.flow_speed)))

        fabric = Fabric(path, surface_charge_density=1e-7) # 0.01 nC / cm^2
        plt.semilogx(1e6*dp2, 1-fabric.penetration(dp2, 1e-3), '--', c=pl.get_color())

    plt.legend(loc='best', fontsize=8)
    plt.xlabel('particle diameter $d_p$ (\si{\micro\meter})')
    plt.ylabel('collection efficiency')
    plt.xlim([1e-2, 1e2])
    plt.ylim([0, 1])

    plt.show()
