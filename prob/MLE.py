from math import exp
from sympy import symbols, diff, Eq, solve, log, Sum, factorial

# Define the symbols
x, mu, sigma = symbols('x mu sigma')

# Define the likelihood function
L = (1 / (sigma * (2 * 3.14159)**0.5)) * exp(-((x - mu)**2) / (2 * sigma**2))

# Take the natural logarithm of the likelihood function
log_L = log(L)

# Take the derivative of the log likelihood function with respect to mu
d_log_L_mu = diff(log_L, mu)

# Take the derivative of the log likelihood function with respect to sigma
d_log_L_sigma = diff(log_L, sigma)

# Set the derivatives equal to 0 and solve for mu and sigma
mu_hat, sigma_hat = solve([Eq(d_log_L_mu, 0), Eq(d_log_L_sigma, 0)], (mu, sigma))

# Print the maximum likelihood estimates
print(f"The maximum likelihood estimate for mu is {mu_hat[0]}")
print(f"The maximum likelihood estimate for sigma is {sigma_hat[0]}")

