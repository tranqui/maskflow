#!/usr/bin/env python3

import numpy as np, matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import erf

class Distribution:
    def moment(self, n):
        raise NotImplementedError

    def __call__(self, d):
        return self.pdf(d)

    @property
    def pdf(self):
        raise NotImplementedError

    @property
    def cdf(self):
        raise NotImplementedError

    @property
    def ccdf(self):
        return lambda x: 1 - self.cdf(x)

    def average(self, f):
        p = self.pdf
        return integrate.quad(lambda d: p(d)*f(d), 0, 5*self.moment(1))[0]

class LogNormalDistribution(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def moment(self, n):
        return np.exp(n*self.mu + 0.5*(n*self.sigma)**2)

    @property
    def pdf(self):
        return lambda d: np.exp(-0.5*(np.log(d)-self.mu)**2/self.sigma**2) / (d*self.sigma*np.sqrt(2*np.pi))

    @property
    def pdf_log(self):
        return lambda d: np.exp(-0.5*(np.log(d)-self.mu)**2/self.sigma**2) / (self.sigma*np.sqrt(2*np.pi))

    @property
    def pdf_log10(self):
        return lambda d: np.log(10) * self.pdf_log(d)

    @property
    def cdf(self):
        return lambda d: 0.5*(1 + erf( (np.log(d) - self.mu) / (self.sigma*np.sqrt(2)) ))

    def average(self, f, norder=100, nsigma=5):
        if self.sigma == 0: return f(np.exp(self.mu))
        p = self.pdf_log
        return integrate.fixed_quad(lambda x: p(np.exp(x))*f(np.exp(x)),
                                    self.mu - nsigma*self.sigma,
                                    self.mu + nsigma*self.sigma,
                                    n=norder)[0]

class MultimodalLogNormalDistribution(Distribution):
    def __init__(self, weights, mu_list, sigma_list, normalise=True):
        self.weights = weights.copy()
        if normalise: self.weights /= np.sum(self.weights)
        self.modes = [LogNormalDistribution(mu, sigma) for mu, sigma in zip(mu_list, sigma_list)]

    def moment(self, n):
        return np.sum([w * mode.moment(n) for w, mode in zip(self.weights, self.modes)])

    @property
    def pdf(self):
        return lambda d: np.sum([w * mode.pdf(d) for w, mode in zip(self.weights, self.modes)], axis=0)

    @property
    def pdf_log(self):
        return lambda d: np.sum([w * mode.pdf_log(d) for w, mode in zip(self.weights, self.modes)], axis=0)

    @property
    def pdf_log10(self):
        return lambda d: np.sum([w * mode.pdf_log10(d) for w, mode in zip(self.weights, self.modes)], axis=0)

    @property
    def cdf(self):
        return lambda d: np.sum([w * mode.cdf(d) for w, mode in zip(self.weights, self.modes)], axis=0)

    def average(self, f, norder=100, nsigma=5):
        return np.sum([w * mode.average(f, norder, nsigma) for w, mode in zip(self.weights, self.modes)])
