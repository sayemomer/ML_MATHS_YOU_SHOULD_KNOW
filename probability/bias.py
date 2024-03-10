# Given probabilities
P_B1_G = 0.9  # Probability that B1 is great
P_B1_M = 0.1  # Probability that B1 is moderate
P_B1_A = 0.0  # Probability that B1 is awful

# Conditional probabilities
P_B2_G_B1_G = 1    # Probability that B2 is great given B1 is great
P_B2_G_B1_M = 0.1  # Probability that B2 is great given B1 is moderate
P_B2_G_B1_A = 0.2  # Probability that B2 is great given B1 is awful

# Calculating the total probability that B2 is great using total probability theorem
P_B2_G = (P_B2_G_B1_G * P_B1_G) + (P_B2_G_B1_M * P_B1_M) + (P_B2_G_B1_A * P_B1_A)
print
print(f"The probability that B2 is great is {P_B2_G:.2f}")

P_B1_G_B2_G = (P_B2_G_B1_G * P_B1_G) / P_B2_G

print(f"The probability that B1 is great given B2 is great is {P_B1_G_B2_G:.2f}")
