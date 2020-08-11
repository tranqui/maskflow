#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

class RawData:
    n, u = np.genfromtxt('thomson.dat').T

    interpolation = interp1d(n, u, kind='cubic', bounds_error=False)
    extrapolation_fit = lambda n, *p: p[3]*n**2 + p[2]*n**1.5 + p[1]*n + p[0]
    p,_ = curve_fit(extrapolation_fit, n, u, (0, 0, -1.1, 0.5))
    extrapolation = lambda n: RawData.extrapolation_fit(n, *RawData.p)

    n_max = np.max(n)
    def fit(n):
        u = RawData.interpolation(n)
        try:
            u[n > RawData.n_max] = RawData.extrapolation(n[n > RawData.n_max])
        except ValueError:
            if n > RawData.n_max: u = RawData.extrapoation(n)
        return u

standard = lambda n: 0.5*n**2
combinatorial = lambda n: 0.5*n*(n-1)
thomson = lambda n: 2*RawData.fit(n)
