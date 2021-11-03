#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted

from maskflow.fabric import penetration
from maskflow import air

mm_per_cm = 10
polypropylene_density = 0.91 # g / cm^3
polypropylene_density /= mm_per_cm**3 # convert to g / mm^3

layer_length = 2 * mm_per_cm # mm
layer_area = layer_length**2 # mm^2

layer_thickness_uncertainty = 5e-3 # mm
layer_mass_uncertainty = 5e-5 # g
layer_length_uncertainty = 0.5 # mm
layer_area_uncertainty = np.sqrt(2) * layer_length * layer_length_uncertainty # mm^2

class SurgicalMask:
    def __init__(self, layer_df, layer_df_sigma, layer_thicknesses, layer_densities):
        """
        Args:
            layer_df: modal fibre diameter (m) in each layer (container)
            layer_df_sigma: log-normal variation of fibre diameter in each layer (container)
            layer_thicknesses: thickness of each layer (mm) (container)
            layer_densities: density of each layer (g/cm^3) (container)
        """
        self.layer_df = np.array(layer_df)
        self.layer_df_sigma = np.array(layer_df_sigma)
        self.layer_thicknesses = np.array(layer_thicknesses)

        self.layer_densities = np.array(layer_densities)
        self.layer_densities /= mm_per_cm**3 # convert to g / mm^3

        self.layer_volume_fractions = self.layer_densities / polypropylene_density
        self.layer_masses = self.layer_densities * layer_area * self.layer_thicknesses
        self.layer_areal_masses = self.layer_masses / layer_area

        self.layer_areal_masses_uncertainty = self.layer_areal_masses * np.sqrt((layer_area_uncertainty/layer_area)**2 + (layer_mass_uncertainty/self.layer_masses)**2)
        self.layer_densities_uncertainty = self.layer_densities * np.sqrt((self.layer_areal_masses_uncertainty/self.layer_areal_masses)**2 + (layer_thickness_uncertainty/self.layer_thicknesses)**2)
        self.layer_volume_fractions_uncertainty = self.layer_volume_fractions * self.layer_densities_uncertainty / self.layer_densities

    def penetration(self, dp, flow_speed, data_folder, return_error=False, fraction=1):
        p = np.ones(dp.size)
        p_error = np.zeros(dp.size)

        for df, df_sigma, L, alpha, dalpha in zip(self.layer_df,
                                                  self.layer_df_sigma,
                                                  fraction*self.layer_thicknesses,
                                                  self.layer_volume_fractions,
                                                  self.layer_volume_fractions_uncertainty):
            data_paths = natsorted(glob('%s/alpha=*_flow=%.3f_df=%.1f_s=%.2f.yml' % (data_folder, flow_speed, 1e6*df, df_sigma)))

            if return_error:
                layer_p, layer_err = penetration(data_paths, dp, flow_speed, df, df_sigma, L, alpha,
                                                 thickness_error=layer_thickness_uncertainty,
                                                 alpha_error=dalpha, return_error=True)
                p_error += (layer_err / layer_p)**2
            else:
                layer_p = penetration(data_paths, dp, flow_speed, df, df_sigma, L, alpha)

            p *= layer_p

        if return_error: return p, p_error
        else: return p

    def pressure_drop(self, flow_speed, T=293):
        pressure_drop = 0

        for df, L, alpha in zip(self.layer_df, self.layer_thicknesses, self.layer_volume_fractions):
            # f = 64*alpha**1.5 * (1 + 56*alpha**3)
            K = -0.5*np.log(alpha) - 0.75 + alpha - 0.25*alpha**2
            f = 16*alpha / K
            dpdz = air.dynamic_viscosity(T) * flow_speed * f / df**2

            pressure_drop += dpdz * (1e-3*L)

        return pressure_drop

# Numbering scheme for surgical masks is taken from:
# Robinson *et al* "Efficacy of face coverings in reducing transmission of COVID-19: calculations based on models of droplet capture", Physics of Fluids 33, 043112 (2021).

SurgicalMask4 = SurgicalMask([2.6e-6, 16.1e-6, 18.1e-6],
                             [0.41, 0.05, 0.06],
                             [0.430, 0.294, 0.214],
                             [0.051296568986254, 0.070047474147296, 0.082496827045114])

SurgicalMask5 = SurgicalMask([2.3e-6, 17.5e-6, 20.7e-6],
                             [0.50, 0.07, 0.08],
                             [0.240, 0.241, 0.358],
                             [0.080457107559262, 0.094489561543825, 0.062535364104949])

if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_folder = '%s/surgicalmaskdata' % dir_path
    flow_speed = 0.063

    D = np.geomspace(1e-8, 1e-5, 101)
    plt.semilogx(D, 1-SurgicalMask4.penetration(D, flow_speed, data_folder), label='SM4')
    plt.semilogx(D, 1-SurgicalMask5.penetration(D, flow_speed, data_folder), label='SM5')

    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.xlabel('particle diameter (m)')
    plt.ylabel('filtration')
    plt.show()
