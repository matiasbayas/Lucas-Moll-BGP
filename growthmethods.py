import numpy as np
from numba import njit
np.seterr(divide='ignore', invalid='ignore')


def agrid(amax, N, amin=0, pivot=0.25):
    """Grid with a+pivot evenly log-spaced between
    amin+pivot and amax+pivot.
    """
    a = np.geomspace(amin+pivot, amax+pivot, N) - pivot
    a[0] = amin  # make sure *exactly* equal to amin
    return a


def updateV(v0, f0, F0, gamma, xgrid, deltax, I, rho, theta, k,
            eta, alpha0, xi,  maxit_v, tol_v):

    alpha = np.empty_like(xgrid)

    for it in range(maxit_v):

        S = np.empty_like(xgrid)

        # Compute S
        for i in range(I):

            diff = v0 - v0[i]
            S[i] = np.sum(diff[i:I]*f0[i:I]*deltax[i:I])

        # Get optimal sigma and alpha:
        sigmau = ((S*alpha0*eta)/xgrid)**(1/(1-eta))
        sigma = np.maximum(np.minimum(sigmau, 1), 0)
        alpha = alpha0*sigma**eta

        # Form matrices:
        C = alpha[:, np.newaxis]@(np.multiply(f0, deltax)[np.newaxis, :])
        C = np.triu(C)

        B = np.zeros([I, I])
        for j in range(1, I):
            B[j, j] = rho - gamma + alpha[j] * (1-F0[j]) + gamma * xgrid[j] / deltax[j]
            B[j, j-1] = - gamma * xgrid[j] / deltax[j]
        B[0, 0] = rho - gamma + alpha[0]*(1-F0[0])

        A = B - C
        b = (1-sigma)*xgrid

        # Solve linear system to obtain value function:
        v = np.linalg.solve(A, b)

        error = np.max(np.max(np.abs(v - v0)))

        if error < tol_v:
            break

        v0 = v

    return v0, sigma, alpha


def updateW(w0, f0, F0, gamma, xgrid, deltax, I, rho, theta, k,
            eta, alpha0, xi, maxit_v, tol_v):

    alpha = np.empty_like(xgrid)

    for it in range(maxit_v):

        S = np.empty_like(xgrid)

        # Compute S
        for i in range(I):

            diff = w0 - w0[i]
            S[i] = np.sum(diff[i:I]*f0[i:I]*deltax[i:I])

        # Get optimal sigma and alpha:
        siu = ((S*alpha0*eta)/xgrid)**(1/(1-eta))
        si = np.maximum(np.minimum(siu, 1), 0)
        alpha = alpha0*si**eta

        # Careful here, have to form slightly different matrices:

        # Form matrices:
        C = alpha[:, np.newaxis]@((f0*deltax)[np.newaxis, :])
        C = np.triu(C)

        D = np.ones([I, 1])@(alpha*f0*deltax[np.newaxis, :])
        D = np.tril(D)

        B = np.zeros([I, I])
        for j in range(1, I):
            B[j, j] = rho - gamma + alpha[j] * \
                (1-F0[j]) + gamma * xgrid[j] / deltax[j] - np.sum(alpha[0:j]*f0[0:j]*deltax[0:j])
            B[j, j-1] = - gamma * xgrid[j] / deltax[j]
        B[0, 0] = rho - gamma + alpha[0]*(1-F0[0]) - alpha[0] * f0[0] * deltax[0]

        A = B - C + D
        b = (1-si)*xgrid

        # Solve linear system to obtain value function:
        w = np.linalg.solve(A, b)

        error = np.max(np.max(np.abs(w - w0)))

        if error < tol_v:
            break

        w0 = w

    return w0, si, alpha


@njit
def updateDist(gamma, alpha, xgrid, deltax, I, rho, theta, k,
               eta, alpha0, xi, maxit_g, tol_g):

    # Initialize to store:
    f = np.empty_like(xgrid)
    F = np.empty_like(xgrid)
    psi = np.empty_like(xgrid)

    for it_g in range(maxit_g):

        # Solve system by iterating backward:

        # Boundary conditions:
        f[I-1] = (k / theta) * xgrid[I-1] ** (-1/theta - 1)
        F[I-1] = 1 - k*xgrid[I-1]**(-1/theta)
        psi[I-1] = gamma/theta

        # Backward iteration:
        for i in range(I-1, 0, -1):
            F[i-1] = F[i] - f[i]*deltax[i]
            psi[i-1] = psi[i] - alpha[i]*f[i]*deltax[i]
            f[i-1] = f[i] + (deltax[i]/(gamma*xgrid[i])) * \
                (f[i]*psi[i] - alpha[i]*f[i]*(1-F[i]) + f[i]*gamma)

        # Update gamma:
        gammanew = xi*(theta * (alpha.T@(f*deltax) +
                                alpha[I-1]*(1-F[I-1]) + alpha[0]*F[0])) + (1-xi)*gamma
        gammanew = np.minimum(theta*np.max(alpha), gammanew)
        error_gamma = np.abs(gammanew - gamma)

        if error_gamma < tol_g:
            break

        gamma = gammanew

    return gamma, f, F


def KGAT(rho, theta, k, eta, alpha0, xi, xmin, xmax, I,
         maxit=20, maxit_g=100, maxit_v=100, tol=1e-7, tol_v=1e-7, tol_g=1e-7):

    xgrid = agrid(xmax, I, xmin)

    deltax = np.zeros(I)
    for i in range(1, I):
        deltax[i] = xgrid[i] - xgrid[i-1]

    # Initial guesses:
    gamma = 0.02
    v0 = xgrid/(rho-gamma)
    F0 = np.exp(-k*xgrid**(-1/theta))
    f0 = (k/theta)*xgrid**(-1/theta-1)*np.exp(-k*xgrid**(-1/theta))
    f0[0] = 0

    for it in range(maxit):

        v0, sigma, alpha = updateV(v0, f0, F0, gamma, xgrid, deltax, I, rho,
                                   theta, k, eta, alpha0, xi,  maxit_v, tol_v)

        gamma, f, F = updateDist(gamma, alpha, xgrid, deltax, I, rho,
                                 theta, k, eta, alpha0, xi, maxit_g, tol_g)

        error = np.max(np.max(np.abs(f - f0)))
        if error < tol:
            print('Convergence OK')
            break

        f0 = f
        F0 = F

    return v0, sigma, alpha, f0, F0, gamma, xgrid, it


def KGATpl(rho, theta, k, eta, alpha0, xi, xmin, xmax, I, maxit=20,
           maxit_g=100, maxit_v=100, tol=1e-7, tol_v=1e-7, tol_g=1e-7):

    xgrid = agrid(xmax, I, xmin)

    deltax = np.zeros(I)
    for i in range(1, I):
        deltax[i] = xgrid[i] - xgrid[i-1]

    # Initial guesses:
    gamma = 0.02
    w0 = xgrid/(rho-gamma)
    F0 = np.exp(-k*xgrid**(-1/theta))
    f0 = (k/theta)*xgrid**(-1/theta-1)*np.exp(-k*xgrid**(-1/theta))
    f0[0] = 0

    for it in range(maxit):

        w0, sigma, alpha = updateW(w0, f0, F0, gamma, xgrid, deltax, I, rho,
                                   theta, k, eta, alpha0, xi,  maxit_v, tol_v)

        gamma, f, F = updateDist(gamma, alpha, xgrid, deltax, I, rho,
                                 theta, k, eta, alpha0, xi, maxit_g, tol_g)

        error = np.max(np.max(np.abs(f - f0)))
        if error < tol:
            print('Convergence OK')
            break

        f0 = f
        F0 = F

    return w0, sigma, alpha, f0, F0, gamma, xgrid, it
