#!/usr/bin/env python3

import sys
import numpy as np
from scipy import integrate, interpolate

def penetration(lamb, L, df, alpha):
    """Mean-field penetration through a filter from filtration theory.

    Args:
        lamb: lambda parameter defining single-fibre efficiency
        L: filter thickness
        df: fibre diameter
        alpha: volume fraction
    """
    return np.exp(-(4*alpha * lamb * L) / (np.pi * df**2 * (1 - alpha)))

class KuwabaraFlowField:
    def __init__(self, alpha):
        self.alpha = alpha
        self.hydrodynamic_factor = -0.5*np.log(alpha) - 0.75 + alpha - 0.25*alpha**2
        self.outer_boundary = 1/np.sqrt(alpha)

        self.A = 0.5 * (1 - 0.5*alpha)
        self.B = 0.5 * (alpha - 1)
        self.C = -0.25 * alpha
        self.D = 1

    @property
    def l(self):
        """Short-hand for the size of the outer boundary (in units of fibre radius)."""
        return self.outer_boundary

    def f(self, r):
        """Radial part of streamfunction."""
        with np.errstate(divide='ignore'):
            return (self.A/r + self.B*r + self.C*r**3 + self.D*r*np.log(r)) / self.hydrodynamic_factor

    def fp(self, r):
        """First (radial) derivative of radial part of streamfunction."""
        with np.errstate(divide='ignore'):
            return (-self.A/r**2 + self.B + 3*self.C*r**2 + self.D*(1 + np.log(r))) / self.hydrodynamic_factor

    def fpp(self, r):
        """Second (radial) derivative of radial part of streamfunction."""
        with np.errstate(divide='ignore'):
            return (2*self.A/r**3 + 6*self.C*r + self.D/r) / self.hydrodynamic_factor

    def u(self, r, th):
        """Radial component of flow field."""
        return np.cos(th) * self.f(r)/r

    def v(self, r, th):
        """Angular component of flow field."""
        return -np.sin(th) * self.fp(r)

    def u_cartesian(self, r, th):
        """Cartesian components of flow field."""
        u = self.u(r, th)
        v = self.v(r, th)
        ux = u*np.cos(th) - v*np.sin(th)
        uy = u*np.sin(th) + v*np.cos(th)
        return ux, uy

    def u_cartesian2(self, x, y):
        """Cartesian components of flow field from cartesian position."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return self.u_cartesian(r, theta)

    def u_perturb(self, r, th, St):
        """Radial component of particle velocity when treating inertia as a perturbation."""
        return self.u(r, th) + St * (self.f(r) - r*self.fp(r)) * (self.f(r)*np.cos(th)**2 - r*self.fp(r)*np.sin(th)**2) / r**3

    def v_perturb(self, r, th, St):
        """Angular component of velocity when treating inertia as a perturbation."""
        return self.v(r, th) + St * np.sin(th) * np.cos(th) * (self.f(r)*(self.fp(r)+r*self.fpp(r)) - r*self.fp(r)**2) / r**2

    def interception_lambda(self, R):
        """Filtration efficiency of single fibre due to interception alone.

        Args:
            R: ratio of particle diameter to fibre diameter
        Returns:
            lambda parameter in units of fibre radius
        """
        return 2*self.f(1+R)

    def stechkina_lambda(self, R, St):
        """Filtration efficiency of single fibre due to interception and inertial impaction
        according to analytic formula of Stechkina (1969).

        This is probably not a very accurate expression.

        Args:
            R: ratio of particle diameter to fibre diameter
            St: stokes number characterising inertia
        Returns:
            lambda parameter in units of fibre radius
        """

        # Hinds (1999) suggests to take the maximum value as a threshold, as the above expression for J is non-monotonic.
        Rmax = 0.00438416 * (59.2 - 56*self.alpha**0.62)**1.25
        Jmax = ((29.6 - 28*self.alpha**0.62) * Rmax**2 - 27.5 * Rmax**2.8)
        J_func = lambda r: ((29.6 - 28*self.alpha**0.62) * r**2 - 27.5 * r**2.8) if r < Rmax else Jmax
        J_func = np.vectorize(J_func)

        impaction = St / (2*self.hydrodynamic_factor**2) * J_func(R)
        return self.interception_lambda(R) + impaction

    def lambda_from_theta0(self, theta0):
        """Determine single-fibre efficiency from the starting angle of the limiting trajectory.

        Returns:
            lambda parameter in units of fibre radius
        """
        return 2*self.f(self.l)*np.sin(theta0)

    def perturbative_lambda_from_theta0(self, theta0, St):
        """Returns lambda in units of fibre radius.

        Initial conditions are slightly different when treating inertia as a perturbation to
        the flow field, so this requires a new function.
        """
        return self.lambda_from_theta0(theta0) + St * ((self.f(self.l) - self.l*self.fp(self.l))* (self.l*self.fp(self.l)*(2*np.pi - 2*theta0 + np.sin(2*theta0)) + self.f(self.l) * (-2*np.pi + 2*theta0 + np.sin(2*theta0)))) / (4*self.l**2)

    def perturbative_theta1(self, R, St, eps=1e-12):
        """Find angle of contact with fibre in the perturbative treatment of inertia.

        This angle can be calculated analytically giving the final condition for the limiting
        trajectory, avoiding an optimisation procedure.
        """

        a = St * (-2*(1 + R)*self.alpha + ((1 + R)**3*self.alpha**2)/2. + (-4 + 2*self.alpha)/(1 + R)**3 + (2 + 2*self.alpha - self.alpha**2)/(1 + R) + (2 - 2*self.alpha + self.alpha**2/2.)/ (1 + R)**5)
        b = -1 + (1 - self.alpha/2.)/(1 + R)**2 + self.alpha - ((1 + R)**2*self.alpha)/2. + 2*np.log(1 + R)
        c = St * (-((1 + R)**3*self.alpha**2) + (-4*self.alpha + self.alpha**2 - 4*np.log(1 + R))/(1 + R) + (2*self.alpha - self.alpha**2 + 4*np.log(1 + R) - 2*self.alpha*np.log(1 + R))/ (1 + R)**3 + (1 + R)* (2*self.alpha + self.alpha**2 + 2*self.alpha*np.log(1 + R)))

        with np.errstate(divide='ignore', invalid='ignore'):
            x = (-b + np.sqrt(b**2 + 8*c*(c-a)))/c
            y = np.sqrt( (2*b*x + 8*(a+c))/c )
            theta = np.arctan2(y,x)
            small_theta = np.pi - np.arctan(np.sqrt(2)/np.sqrt(-1 + np.sqrt(1 + 256*St**2*(-1 + self.alpha)**2)))
            try:
                theta[R < eps] = small_theta[R < eps]
            except:
                if R < eps: theta = small_theta

        try:
            theta[np.isnan(theta)] = 0.5*np.pi
        except:
            if np.isnan(theta): theta = 0.5*np.pi

        return theta

    def perturbative_find_trajectory(self, R, St, r0, th0, reverse=False, max_step=1e-3):
        th1 = np.pi if reverse else 0

        drdth = lambda t,x,St: np.array([x*self.u_perturb(x,t,St)/self.v_perturb(x,t,St)]).T

        penetrate = lambda t,x,St: self.l - x
        penetrate.terminal = True
        penetrate.direction = -1
        if reverse:
            events = (penetrate,)
        else:
            collide = lambda t,x,St: x - (1+R)
            collide.terminal = True
            collide.direction = -1
            events = (penetrate,collide)

        with np.errstate(divide='ignore', invalid='ignore'):
            trajectory = integrate.solve_ivp(drdth, [th0, th1], [r0], args=(St,),
                                             events=events, max_step=max_step,
                                             vectorized=True, dense_output=True)

        return trajectory

    def perturbative_limit_trajectory(self, R, St):
        # integrate backwards from the final point
        th1 = self.perturbative_theta1(R, St)
        trajectory = self.perturbative_find_trajectory(R, St, 1+R, th1, reverse=True)
        th0 = trajectory.t[-1]
        return trajectory, th0, th1, self.perturbative_lambda_from_theta0(th0, St)

    def perturbative_impaction_efficiency(self, R, St, return_angles=False):
        limit_trajectory = self.perturbative_limit_trajectory(R, St)
        return limit_trajectory[1:] if return_angles else limit_trajectory[-1]

    def find_trajectory_full(self, R, St, r0, th0, tmax=100, max_step=1, **kwargs):
        if St == 0:
            dxdt2 = lambda r,th: np.array([self.u(r,th), self.v(r,th)/r])
            dxdt = lambda t,x: dxdt2(*x)

            penetrate = lambda t,x: self.l - x[0]
            penetrate.terminal = True
            penetrate.direction = -1
            events = (penetrate,)

            with np.errstate(divide='ignore', invalid='ignore'):
                x0 = [r0, th0]
                trajectory = integrate.solve_ivp(dxdt, [0, tmax], x0, #jac=jacobian,
                                                 events=events, max_step=max_step,
                                                 vectorized=True, dense_output=True,
                                                 **kwargs)

            return trajectory

        dudt = lambda r,th,ur,ut: -(ur - self.u(r,th))/St + ut**2/r
        dvdt = lambda r,th,ur,ut: -(ut - self.v(r,th))/St - ur*ut/r
        duvdt = lambda r,th,ur,ut: [dudt(r,th,ur,ut), dvdt(r,th,ur,ut)]
        dxdt2 = lambda r,th,ur,ut: np.array([ur, ut/r, *duvdt(r,th,ur,ut)])
        dxdt = lambda t,x: dxdt2(*x)

        # Jacobians are needed for some solvers, but not for RK45 (the current/default solver)
        # jacobian = lambda r,th,ur,ut,f,fp,fpp: \
        #     np.array([[0, 0, 1, 0],
        #               [-ut/r**2, 0, 0, 1/r],
        #               [-(ut/r)**2 + np.cos(th)*(fp - f/r)/r/St, -np.sin(th)*f/r/St, -1/St, 2*ut/r],
        #               [ur*ut/r**2 - np.sin(th)*fpp/St, -np.cos(th)*fp/St, -ut/r, -ur/r - 1/St]])
        # jacobian = lambda t,x: jacobian2(*x, self.f(x[0]), self.fp(x[0]), self.fpp(x[0]))

        r = lambda r,th,ur,ut: r
        theta = lambda r,th,ur,ut: th
        penetrate = lambda t,x: self.l - r(*x)
        collide = lambda t,x: r(*x) - (1+R)
        wrong_direction = lambda t,x: x[2]*self.u(x[0],x[1]) + x[3]*self.v(x[0],x[1]) + 0.9

        penetrate.terminal = True
        penetrate.direction = -1
        collide.terminal = True
        collide.direction = -1
        wrong_direction.terminal = True
        wrong_direction.direction = -1
        events = (penetrate,collide,wrong_direction)

        with np.errstate(divide='ignore', invalid='ignore'):
            x0 = [r0, th0, self.u(r0,th0), self.v(r0,th0)]
            trajectory = integrate.solve_ivp(dxdt, [0, tmax], x0, #jac=jacobian,
                                             events=events, max_step=max_step,
                                             vectorized=True, dense_output=True,
                                             **kwargs)

        return trajectory

    def does_collide(self, R, St, th0, return_trajectory=False, tmax=100, max_step=1):
        trajectory = self.find_trajectory_full(R, St, self.l, th0, tmax, max_step)
        collided = len(trajectory.t_events[1]) > 0
        return (collided, trajectory) if return_trajectory else collided

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    R = 0.2
    alpha = 0.15
    flow = KuwabaraFlowField(alpha)

    plt.figure()
    plt.gca().add_patch(Circle((0,0), 1+R, fill=False, color='k', ls='dashed', lw=1))
    plt.gca().add_patch(Circle((0,0), 1, color='r', lw=0))

    first = True
    print('St\tth0    th1    lam')
    for St in [0.01, 0.1, 0.5, 1]:
        traj,th0,th1,lam = flow.perturbative_limit_trajectory(R, St)
        theta, rho = traj.t, traj.y.reshape(-1)
        x1 = rho*np.cos(theta)
        y1 = rho*np.sin(theta)

        trajectory = flow.find_trajectory_full(R, St, flow.outer_boundary, th0,
                                               tmax=1e2, max_step=0.1)
        rho, theta, _, _ = trajectory.y
        x2 = rho*np.cos(theta)
        y2 = rho*np.sin(theta)

        label = '%.1g' % St
        if first: label = '$\mathrm{St}=%s$' % label
        pl, = plt.plot(x2, y2, '-', label=label)
        plt.plot(x1, y1, '--', c=pl.get_color())

        suffix = ' (interception=%.4f)' % flow.interception_lambda(R) if first else ''
        print('%.2g\t%.4f %.4f %.4f %s' % (St, th0, th1, lam, suffix))
        first = False

    plt.legend(loc='lower left', fontsize=8)
    plt.xlabel('$x/a_f$')
    plt.ylabel('$y/a_f$')

    plt.show()
