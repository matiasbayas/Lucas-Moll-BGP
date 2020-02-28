import numpy as np
import scipy as sp
from scipy.stats import norm

"""Part 1: defining exogenous grids and states.

Nothing here very performance-sensitive, so we strive
for clarity."""

def agrid(amax,N,amin=0,pivot=0.25):
    """Grid with a+pivot evenly log-spaced between
    amin+pivot and amax+pivot.
    """
    a = np.geomspace(amin+pivot,amax+pivot,N) - pivot
    a[0] = amin # make sure *exactly* equal to amin
    return a

def variance(x, pi):
    """Variance of x distributed with density pi"""
    return np.sum(pi*(x-np.sum(pi*x))**2)

def stationary(Pi, pi_seed=None, tol=1E-11, maxit=10_000):
    """Iterate to find invariant distribution of Markov chain.
    
    Parameters
    ----------
    Pi : array (S*S), Markov chain
    pi_seed : [optional] array (S), seed for iteration
    tol : [optional] float, stop if iterations change less
           than tol in Linfty norm
    maxit : [optional] int, stop if more than maxit iterations
    
    Returns
    ----------
    pi : array (S), invariant distribution of Markov chain
    """
    if pi_seed is None:
        pi = np.ones(Pi.shape[0])/Pi.shape[0]
    else:
        pi = pi_seed
        
    for it in range(maxit):
        pi_new = pi @ Pi
        if np.max(np.abs(pi_new-pi)) < tol:
            break
        pi = pi_new 
    pi = pi_new
    
    return pi

def markov_tauchen(rho, sigma, N=7, m=3):
    """Tauchen method discretizing AR(1) s_t = rho*s_(t-1) + eps_t.
        
    Parameters
    ----------
    rho : float, persistence
    sigma : float, unconditional sd of s_t
    N : int, number of states in discretized Markov process
    m : float, discretized s goes from approx -m*sigma to m*sigma
    
    Returns
    ----------
    s : array (N), states in discretized process
    pi : array (N), stationary distribution of discretized process
    Pi : array (N*N), Markov matrix for discretized process
    y : array (N), states proportional to exp(s) s.t. E[y] = 1
    """
    
    # make normalized grid, start with cross-sectional sd of 1
    s = np.linspace(-m, m, N)
    ds = s[1] - s[0]
    sd_innov = np.sqrt(1-rho**2)
    
    # standard Tauchen method to generate Pi given N and m
    Pi = np.empty((N, N))
    Pi[:, 0] = norm.cdf(s[0] - rho*s + ds/2, scale=sd_innov)
    Pi[:, -1] = 1 - norm.cdf(s[-1] - rho*s - ds/2, scale=sd_innov)
    for j in range(1, N - 1):
        Pi[:, j] = (norm.cdf(s[j] - rho*s + ds/2, scale=sd_innov) - 
                    norm.cdf(s[j] - rho*s - ds/2, scale=sd_innov))
        
    # invariant distribution and scaling
    pi = stationary(Pi)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s)/np.sum(pi*np.exp(s))
    
    return s, pi, Pi, y


def markov_rouwenhorst(rho, sigma, N=7):
    """Rouwenhorst method analog to markov_tauchen"""
    
    # parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])

    # implement recursion to build from n=3 to n=N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * Pi
        P2[:-1, 1:] = (1 - p) * Pi
        P3[1:, :-1] = (1 - p) * Pi
        P4[1:, 1:] = p * Pi
        Pi = P1 + P2 + P3 + P4
        Pi[1:-1] /= 2
        
    # invariant distribution and scaling
    pi = stationary(Pi)
    s = np.linspace(-1, 1, N)
    s *= (sigma / np.sqrt(variance(s, pi)))
    y = np.exp(s)/np.sum(pi*np.exp(s))
    
    return s, pi, Pi, y

"""Part 2: backward iteration using endogenous gridpoints.

Much more performance-sensitive, so we define custom compiled
functions for interpolation, etc.
"""

# our primary tool for easy, efficient compiled code is Numba
from numba import jit, njit, vectorize, guvectorize

# Numba's guvectorize decorator compiles and allows function to be
# automatically broadcast by NumPy when dimensions differ
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], 
             '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.
    
    Complexity O(n+nq), so most efficient when x and xq have
    comparable number of points. Extrapolates linearly when
    xq out of domain of x.
    
    Parameters
    ----------
    x : array (n) ascending data points
    xq : array (nq) ascending query points
    y : array (n) data points
    yq : array (nq) empty to be filled with interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]
    
    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx-2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi+1]

        xqpi_cur = (x_high-xq_cur) / (x_high-x_low)
        yq[xqi_cur] = xqpi_cur*y[xi] + (1-xqpi_cur)*y[xi+1]


# Numba's njit decorator does just-in-time compilation of function,
# short for jit(nopython=True), which throws error if Numba cannot
# successfully compile everything
@njit
def setmin(x, xmin):
    """Set 2-dimensional array x where each row is ascending
    equal to equal to max(x, xmin)."""
    ni, nj = x.shape
    for i in range(ni):
        for j in range(nj):
            if x[i,j] < xmin:
                x[i,j] = xmin
            else:
                break
                
def backward_iterate(Pi_p, uc_p, a_grid, a_grid_p, y, r, r_post, beta, eis=1):
    """Backward iteration using endogenous gridpoints for household
    with one exogenous state s and one endogenous asset state a, with
    CRRA utility.
    
    State space has dimensions S*A.
    
    Parameters
    ----------
    Pi_p : array (S*S), Markov matrix for s tomorrow
    uc_p : array (S*A), marginal utility tomorrow
    a_grid : array (A), asset grid today
    a_grid_p : array (A), asset grid tomorrow
    y : array (S), income levels today
    r : scalar, real interest rate in Euler equation today
    r_post: scalar, ex-post real interest rate on yesterday's assets
    beta: scalar or array (S*1), discount rate today
    eis: [optional] scalar, EIS in CRRA utility
    
    Returns
    ----------
    uc : array (S*A), marginal utility today
    a : array (S*A), asset policy today
    """
    
    # take expectations over Markov s and discount to get marginal
    # utility today on today's grid for s but TOMORROW's grid for a
    uc_nextgrid = ((1+r)*beta*Pi_p) @ uc_p
    
    # implied consumption
    c_nextgrid = uc_nextgrid **(-eis)
    
    # cash on hand at beginning of period
    coh = (1+r_post)*a_grid + y[:, np.newaxis]
    
    # we have consumption c today for each a' tomorrow, so we have
    # map from coh=c+a' to a' on tomorrow's asset grid.
    # interpolate in coh to get map of coh to a' on today's grid.
    # do this separately for each s (automatic via broadcasting).
    a = interpolate_y(c_nextgrid + a_grid_p, coh, a_grid_p)
    
    # set constrained agents' policies to minimum asset level
    setmin(a, a_grid_p[0])
    
    # calculate consumption, return marginal utility and asset policy
    c = coh - a
    uc = c ** (-1/eis)
    return uc, a

@njit
def within_tolerance(x1, x2, tol):
    """Efficiently test max(abs(x1-x2)) <= tol for same-dim x1, x2"""
    # implement by obtaining flattened views using ravel, then looping
    y1 = x1.ravel()
    y2 = x2.ravel()
    
    for i in range(y1.shape[0]):
        if np.abs(y1[i] - y2[i]) > tol:
            return False
    return True

def pol_ss(Pi, a_grid, y, r, beta, eis=1, uc_seed=None,
           tol=1E-8, maxit=5000):
    """Backward iterate to find ss asset policy and marg utility.
    
    Note that most parameters except the last 3 are analogs of
    backward_iterate, and also return values are the same.
    
    Parameters
    ----------
    Pi : array (S*S), steady-state Markov matrix
    a_grid : array (A), steady-state asset grid
    y : array (S*A), steady-state income levels
    r : scalar, steady-state real interest rate
    beta : scalar or array (S*1), steady-state discount rate
    eis : [optional] scalar, EIS in CRRA utility
    uc_seed : [optional] array (S*A), seed for marginal utility
    tol : [optional] float, stop if iterations change policy less
           than tol in Linfty norm
    maxit : [optional] int, stop if more than maxit iterations
    
    Returns
    ----------
    uc : array (S*A), steady-state marginal utility
    a : array (S*A), steady-state asset policy
    """
    if uc_seed is None:
        # if no uc_seed, start by assuming 10% of coh consumed
        uc = (0.1*((1+r)*a_grid + y[:, np.newaxis])) ** (-1/eis)
    else:
        uc = uc_seed

    # iterate until convergence of a policy by tol, or maxit
    for it in range(maxit):
        uc, anew = backward_iterate(Pi, uc, a_grid, a_grid, y, r,
                                    r, beta, eis)
        
        # only check convergence every 5 iterations for efficiency
        if it % 5 == 1 and within_tolerance(a, anew, tol):
            break
        a = anew
    else:
        raise ValueError(f'No convergence after {maxit} backward iterations!')
    a = anew
    
    return uc, a

"""Part 3: tools for forward iteration on the s*a distribution,
including finding a stationary distribution."""

# for speed and memory savings, we assign coordinates to uint32,
# an unsigned 32-bit integer (max ~4 billion)
@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'],
             '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """Efficient linear interpolation exploiting monotonicity like
    interpolate_rule and with nearly-identical code, but rather than
    giving values of y at query points, gives interpolated
    "coordinates" xqi and xqpi:
    
    xq = xqpi*x[xqi] + (1-xqpi)*x[xqi+1]
    
    xqpi is in interval (0,1] except when xq outside of domain of x.
    """
    nxq, nx = xq.shape[0], x.shape[0]
    
    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx-2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi+1]

        xqpi[xqi_cur] = (x_high-xq_cur) / (x_high-x_low)
        xqi[xqi_cur] = xi
    
@njit   
def forward_iterate(D, Pi_T, a_pol_i, a_pol_pi):
    """Update distribution of agents over state space S*A by
    iterating forward asset policy, then state.
    
    Asset policy is represented using interpolated coordinates
    a_pol_i and a_pol_pi: when an agent tries to choose a level
    of assets in between two gridpoints, we update the
    distribution as if she chose a lottery between those two
    gridpoints that gives the same level of assets on average.
    
    Parameters
    ----------
    D : array (S*A), beginning-of-period distribution s_t, a_(t-1)
    Pi_T : array (S*S), transpose Markov matrix for s_t to s_(t+1)
    a_pol_i : int array (S*A), left gridpoint of asset policy
    a_pol_pi : array (S*A), share left gridpoint in asset policy
    
    Returns
    ----------
    Dnew : array (S*A), beginning-of-next-period dist s_(t+1), a_t
    """
    
    # first create Dnew from updating asset state
    Dnew = np.zeros_like(D)
    for s in range(D.shape[0]):
        for i in range(D.shape[1]):
            apol = a_pol_i[s,i]
            api = a_pol_pi[s,i]
            d = D[s,i]
            Dnew[s, apol] += d*api
            Dnew[s, apol+1] += d*(1-api)
    
    # then use transpose Markov matrix to update 's'
    Dnew = Pi_T @ Dnew
    
    return Dnew

def dist_ss(Pi, a_grid, a_pol,
            D_seed=None, pi_seed=None, tol=1E-10, maxit=10_000):
    """Iterate to find steady-state distribution of s_t, a_(t-1).
    
    Parameters
    ----------
    Pi : array (S*S), steady-state Markov matrix
    a_grid : array (A), steady-state asset grid
    a_pol : array (S*A), steady-state asset policy function
    D_seed : [optional] array (S*A), seed for s*a distribution
    pi_seed : [optional] array (S), seed for s distribution
    tol : [optional] float, stop if iterations change less
           than tol in Linfty norm
    maxit : [optional] int, stop if more than maxit iterations
    
    Returns
    ----------
    D : array (S*A), steady-state s*a distribution of agents
    """
    
    if D_seed is None:
        # compute separately stationary dist of s, to start there
        # assume a uniform distribution on assets otherwise
        pi = stationary(Pi, pi_seed)
        D = np.tile(pi[:, np.newaxis],
                    (1, a_grid.shape[0])) / a_grid.shape[0]
    else:
        D = D_seed
    
    # obtain interpolated-coordinate asset policy rule
    a_pol_i, a_pol_pi = interpolate_coord(a_grid, a_pol)
    
    # to make matrix multiplication more efficient, make
    # separate copy of Pi transpose so memory right alignment
    Pi_T = Pi.T.copy()
    
    # iterate until convergence by tol, or maxit
    for it in range(maxit):
        Dnew = forward_iterate(D, Pi_T, a_pol_i, a_pol_pi)
        
        # only check convergence every 5 iterations for efficiency
        if it % 5 == 0 and within_tolerance(D, Dnew, tol):
            break
        D = Dnew
    else:
        raise ValueError(f'No convergence after {maxit} forward iterations!')
    D = Dnew
    
    return D

"""Part 4: put together backward and forward iteration to solve ss
   household problem"""

def household_ss(Pi, a_grid, y, r, beta, eis=1,
                 uc_seed = None, D_seed = None, pi_seed = None):
    """Wrapper to backward iteration pol_ss to get steady-state pol,
    dist_ss to get steady-state distribution, reports idiosyncratic
    and aggregate results.
    """

    # backward iterate till convergence
    uc, a = pol_ss(Pi, a_grid, y, r, beta, eis, uc_seed)
    
    # implied c policy (not returned)
    c = (1+r)*a_grid + y[:, np.newaxis] - a
    
    # forward iterate till convergence
    D = dist_ss(Pi, a_grid, a, D_seed, pi_seed)
    
    # return handy dict with results and inputs
    inputs = {'Pi': Pi, 'a_grid': a_grid, 'y': y,
              'r': r, 'beta': beta, 'eis': eis}
    results = {'D': D, 'uc': uc, 'a': a, 'c': c,
               'A': np.vdot(D, a), 'C': np.vdot(D, c)}
    return {**inputs, **results}

"""Part 6: the early fruits of heterogeneity - MPCs"""
def mpcs(c, a_grid, y, r, a):
    """Approximate mpc, with symmetric differences where possible,
    exactly setting mpc=1 for constrained agents"""
    mpcs = np.empty_like(c)
    coh = (1+r)*a_grid + y[:, np.newaxis]
    
    # symmetric differences away from boundaries
    mpcs[:, 1:-1] = (c[:, 2:]-c[:, 0:-2])/(coh[:, 2:]-coh[:, :-2])

    # asymmetric first differences at boundaries
    mpcs[:,0]  = (c[:, 1]-c[:, 0])/(coh[:, 1]-coh[:, 0])
    mpcs[:,-1] = (c[:, -1]-c[:, -2])/(coh[:, -1]-coh[:, -2])

    # special case of constrained
    mpcs[a==a_grid[0]] = 1
    
    return mpcs
