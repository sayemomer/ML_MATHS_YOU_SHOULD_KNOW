import matplotlib.pyplot as plt

# Assuming 'lambdas' is a list of different regularization strengths (λ)
# 'neg_log_likelihoods' is a list of corresponding negative log likelihoods
# 'norms_of_w' is a list of corresponding norms of w.

lambdas = [0.1, 1, 10, 100]  # example values for λ
neg_log_likelihoods = [59.09877948122527,81.7519 ,108.8628 , 16266.0366]  # example values for negative log likelihood
norms_of_w = [5.958620942049719 , 3.439324910449318 , 0.9250969463979295 , 12.735783166540324 ]  # example values for norm of w

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot negative log likelihood
ax1.set_xlabel('Regularization strength (λ)')
ax1.set_ylabel('Negative Log Likelihood', color='tab:blue')
ax1.plot(lambdas, neg_log_likelihoods, marker='o', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xscale('log')  # Use logarithmic scale for regularization strength

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot norm of w on the second y-axis
ax2.set_ylabel('Norm of W', color='tab:red')
ax2.plot(lambdas, norms_of_w, marker='o', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()  # To ensure the right y-label is not slightly clipped
plt.title('Neg. Log Likelihood and Norm of W vs Regularization Strength')
plt.show()
