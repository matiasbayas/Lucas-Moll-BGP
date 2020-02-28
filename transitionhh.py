

# The idea is that you will eventually go to new ss.
# So first solve for the steady state under the new tail parameter:

import growthmethods as gm
import numpy as np
import matplotlib.pyplot as plt

# Final Tail Parameter:
theta1 = 0.7
# Initial Tail Parameter:
theta0 = 0.5

# Other Parameters:
rho = 0.06
k = 0.05
eta = 0.3
alpha0 = 0.0849

xi = 0.4

I = 2001
xmin = 0
xmax = 3

# Compute value function and distribution at the final and initial ss
v1, sigma1, alpha1, f1, F1, gamma1, xgrid1, it1 = gm.KGAT(rho, theta1, k, eta, alpha0, xi, xmin, xmax, I)
v0, sigma0, alpha0, f0, F0, gamma0, xgrid0, it0 = gm.KGAT(rho, theta0, k, eta, alpha0, xi, xmin, xmax, I)


# Process for tail parameter during the transition:
T = 100
rhos = 0.75
thetas = np.zeros(T)
thetas[0] = theta0
thetas[T-1] = theta1
for t in range (1,T):
    thetas[t] = theta1 + rhos*(thetas[t-1]-theta1)

plt.plot(thetas)
plt.show()

plt.plot(xgrid1, sigma1)
plt.plot(xgrid0, sigma0)
plt.show()
