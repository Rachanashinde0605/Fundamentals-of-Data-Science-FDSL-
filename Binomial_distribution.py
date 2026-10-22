import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

# 1. Load dataset
data = pd.read_csv("Titanic_Dataset.csv")

# 2. Define parameters
# Survived column: 1 = Success, 0 = Failure
n = len(data)                      # Number of passengers
p = data["Survived"].mean()        # Probability of survival

print("Total number of passengers:", n)
print("Probability of survival:", p)

# 3. Generate possible number of successes and their probabilities
x = np.arange(0, n + 1)

pmf = binom.pmf(x, n, p)            # Probability Mass Function
cdf = binom.cdf(x, n, p)            # Cumulative Distribution Function

# 4. Compute mean and standard deviation
mean, var = binom.stats(n, p, moments='mv')
std_dev = np.sqrt(var)

print("\nMean (Expected survivors):", mean)
print("Variance:", var)
print("Standard Deviation:", std_dev)

# 5. Plot the Binomial Distribution
plt.figure(figsize=(10, 5))

plt.bar(x, pmf, color='skyblue', edgecolor='black', label='PMF')
plt.axvline(mean, color='red', linestyle='--', label='Mean')

plt.title(f'Binomial Distribution - Titanic Survival (n={n}, p={p:.2f})')
plt.xlabel('Number of Survivors')
plt.ylabel('Probability')
plt.legend()
plt.show()

# 6. Real-world probability questions

# Probability that at least 200 passengers survive
p_at_least_200 = 1 - binom.cdf(199, n, p)
print("\nP(X >= 200):", p_at_least_200)

# Probability that exactly 180 passengers survive
p_exactly_180 = binom.pmf(180, n, p)
print("P(X = 180):", p_exactly_180)

# Probability that at most 200 passengers survive
p_at_most_200 = binom.cdf(200, n, p)
print("P(X <= 200):", p_at_most_200)