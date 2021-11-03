#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
try: from .distributions import MultimodalLogNormalDistribution
except: from distributions import MultimodalLogNormalDistribution

# From Johnson et al (2011):
# DF: average APS sample dilution factors, relating the concentration in the sample to that at the source
# EF: evaporative diameter shrinkage factor, a correction used for APS samples
# SF: dynamic spread factor

class ExhalationMode(MultimodalLogNormalDistribution):
    def __init__(self, weights, *args, **kwargs):
        self.number_density_per_second = np.sum(weights) # cm^-3 s^-1
        super().__init__(weights, *args, **kwargs, normalise=True)

    def __call__(self, d):
        return self.pdf_log(d)

    def hydrated(self, factor):
        mu, sigma = np.array([(mode.mu, mode.sigma) for mode in self.modes]).T
        return ExhalationMode(self.weights, mu + np.log(factor), sigma)

class ExperimentalFits:
    def __init__(self, aerosol_prefactors,
                 aerosol_count_median, aerosol_geometric_stddev,
                 aerosol_evaporation_factor, aerosol_dilution_factor,
                 oral_prefactors=None,
                 oral_count_median=None, oral_geometric_stddev=None,
                 oral_spread_factor=None):
        aerosol_prefactors = [a * aerosol_dilution_factor for a in aerosol_prefactors]
        all_prefactors = aerosol_prefactors.copy()
        all_count_median = aerosol_count_median.copy()
        all_geometric_stddev = aerosol_geometric_stddev.copy()

        exhaled_aerosol_count_median = [a / aerosol_evaporation_factor for a in aerosol_count_median]
        exhaled_all_count_median = exhaled_aerosol_count_median.copy()

        if oral_prefactors is not None:
            exhaled_oral_count_median = [a / oral_spread_factor for a in oral_count_median]

            for p,c1,c2,s in zip(oral_prefactors,
                                 oral_count_median,
                                 exhaled_oral_count_median,
                                 oral_geometric_stddev):
                all_prefactors += [p]
                all_count_median += [c1]
                exhaled_all_count_median += [c2]
                all_geometric_stddev += [s]

            self.measured_oral_droplet_distribution = \
                ExhalationMode(oral_prefactors,
                               np.log(oral_count_median),
                               np.log(oral_geometric_stddev))

            self.exhaled_oral_droplet_distribution = \
                ExhalationMode(oral_prefactors,
                               np.log(exhaled_oral_count_median),
                               np.log(oral_geometric_stddev))

        self.measured_droplet_distribution = \
            ExhalationMode(all_prefactors,
                           np.log(all_count_median),
                           np.log(all_geometric_stddev))

        self.exhaled_droplet_distribution = \
            ExhalationMode(all_prefactors,
                           np.log(exhaled_all_count_median),
                           np.log(all_geometric_stddev))

        self.measured_aerosol_distribution = \
            ExhalationMode(aerosol_prefactors,
                           np.log(aerosol_count_median),
                           np.log(aerosol_geometric_stddev))

        self.exhaled_aerosol_distribution = \
            ExhalationMode(aerosol_prefactors,
                           np.log(exhaled_aerosol_count_median),
                           np.log(aerosol_geometric_stddev))

class Gregson2020:
    dilution_factor = 1 # not accounted for in Gregson2020
    evaporation_factor = 1 # not accounted for in Gregson2020

    TidalBreathing = ExperimentalFits([0.494, 0.266],
                                      [0.55, 1.07],
                                      [1.29, 1.32],
                                      evaporation_factor,
                                      dilution_factor)

    Speaking70to80dB = ExperimentalFits([0.354, 0.1],
                                        [0.5, 1.34],
                                        [1.58, 1.48],
                                        evaporation_factor,
                                        dilution_factor)

    Speaking90to100dB = ExperimentalFits([0.749, 1.223],
                                         [0.53, 1.28],
                                         [1.32, 1.78],
                                         evaporation_factor,
                                         dilution_factor)

    Singing70to80dB = ExperimentalFits([0.395, 0.495],
                                       [0.52, 1.14],
                                       [1.32, 1.7],
                                       evaporation_factor,
                                       dilution_factor)

    Singing90to100dB = ExperimentalFits([1, 2.093],
                                        [0.55, 1.27],
                                        [1.26, 1.82],
                                        evaporation_factor,
                                        dilution_factor)

class Johnson2011:
    dilution_factor = 3.6
    cough_dilution_factor = 4.3
    evaporation_factor = 0.5
    spread_factor = 1.5

    TidalBreathing = ExperimentalFits([0.0175],
                                      [0.8],
                                      [1.275],
                                      evaporation_factor,
                                      dilution_factor)

    Speaking = ExperimentalFits([0.015, 0.019],
                                [0.807, 1.2],
                                [1.30, 1.66],
                                evaporation_factor,
                                dilution_factor,
                                [0.00126], [217], [1.795], spread_factor)

    Coughing = ExperimentalFits([0.021, 0.033],
                                [0.784, 0.8],
                                [1.25, 1.68],
                                evaporation_factor,
                                cough_dilution_factor,
                                [0.01596], [185], [1.837], spread_factor)
