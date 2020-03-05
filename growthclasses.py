import growthmethods as gm
import matplotlib.pyplot as plt


class LucasMoll:

    """
    Class that generates an instance of the Lucas-Moll endogenous growth model
    Also contains methods for solving for the BGP of the model.

    The class takes as parameters:

        - rho           discount rate
        - theta         tail parameter of prod. distribution
        - k             controls position of the tail of distribution
                        (i.e. governs units of relative productivity)
        - eta           curvature of meeting technology function
        - alpha0        constant in meeting technology function
        - xi            numerical updating parameter
        - xmax, xmin    bounds for relative productivity
        - I             size of grid for productivity

    The class also has some methods to compute the  equilibrium BGP of the
    model and to solve for an optimally planned economy:

    1) eqm():       computes the value function and distribution along eqm BGP
    2) planner():   computes the solution to planning problem outlined in
                    Lucas-Moll
    """

    def __init__(self, rho=0.06,
                 theta=0.5,
                 k=0.05,
                 eta=0.3,
                 alpha0=0.0849,
                 xi=0.4,
                 xmin=0,
                 xmax=3,
                 I_x=2001):

        self.rho, self.theta, self.k = rho, theta, k
        self.eta, self.alpha0 = eta, alpha0
        self.xi, self.xmin, self.xmax, self.I_x = xi, xmin, xmax, I_x

    def eqm(self, maxit=50, tol=1e-7):

        rho, theta, k = self.rho, self.theta, self.k
        eta, alpha0 = self.eta, self.alpha0
        xi, xmin, xmax, I_x = self.xi, self.xmin, self.xmax, self.I_x

        self.v, self.sigma, self.alpha, self.f, self.F, self.gamma, self.xgrid, self.it = gm.KGAT(
            rho, theta, k, eta, alpha0, xi, xmin, xmax, I_x)

        return self.v, self.sigma, self.alpha, self.f, self.F, self.gamma, self.xgrid, self.it

    def planner(self, maxit=50, tol=1e-7):

        rho, theta, k = self.rho, self.theta, self.k
        eta, alpha0 = self.eta, self.alpha0
        xi, xmin, xmax, I_x = self.xi, self.xmin, self.xmax, self.I_x

        w, sigmap, alphap, fp, Fp, gammap, xgridp, itp = gm.KGATpl(
            rho, theta, k, eta, alpha0, xi, xmin, xmax, I_x)

        return w, sigmap, alphap, fp, Fp, gammap, xgridp, itp

    def plotresults(self):

        v, f, F, xgrid = self.v, self.f, self.F, self.xgrid
        med = xgrid[[0.499 < i < 0.501 for i in F]]
        index = xgrid <= med*3.5
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        ax1.plot(xgrid[index]/med, v[index], linestyle='-.', color='black')
        ax2.plot(xgrid[index]/med, f[index], linestyle='-.')
        ax3.plot(xgrid[index]/med, F[index], linestyle='-.', color='red')
        ax1.set_title('Value Function - $v(x)$')
        ax2.set_title('Density Function - $f(x)$')
        ax3.set_title('Distribution Function - $F(x)$')
        ax1.set_xlabel('Relative Productivity - $x$')
        ax2.set_xlabel('Relative Productivity - $x$')
        ax3.set_xlabel('Relative Productivity - $x$')
        plt.show()
