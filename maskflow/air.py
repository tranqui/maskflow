#!/usr/bin/env python3

import os
import numpy as np
current_dir = os.path.dirname(os.path.realpath(__file__))

class RawData:
    # data taken from https://www.engineersedge.com/physics/viscosity_of_air_dynamic_and_kinematic_14483.htm
    temp, density, specific_heat, thermal_conductivity, thermal_diffusivity, dynamic_viscosity, kinematic_viscosity, prandtl_number = np.genfromtxt('%s/air-properties.csv' % current_dir).T

    water_freezing = 273.15 # K
    temp += water_freezing

fit_degree = 5
density = np.poly1d(np.polyfit(RawData.temp, RawData.density, fit_degree))
dynamic_viscosity = np.poly1d(np.polyfit(RawData.temp, RawData.dynamic_viscosity, fit_degree))
kinematic_viscosity = np.poly1d(np.polyfit(RawData.temp, RawData.kinematic_viscosity, fit_degree))

boltzmann_constant = 1.38e-23 # J / K
atmospheric_pressure = 101325 # Pa
collisional_cross_section = np.pi * (3.64e-10)**2 # m^2 (using kinetic diameter of nitrogen, from wikipedia)
mean_free_path = lambda T: boltzmann_constant*T / (np.sqrt(2) * collisional_cross_section * atmospheric_pressure) # m
