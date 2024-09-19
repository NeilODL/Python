import numpy as np
import pandas as pd
from scipy.stats import qmc
from fdm import fdm
import time

start_time = time.time()

# Sample size (number of parameter sets to generate)
N = 6000 # Adjust this value as needed

# Define the parameter ranges
param_ranges = {
    'tau': (0.1, 1.4),      # Time to maturity
    'r': (0.0, 0.10),       # Risk-free rate
    'rho': (-0.95, 0.0),    # Correlation
    'kappa': (0.0, 2.0),    # Reversion speed
    'theta': (0.0, 0.5),    # Long average variance
    'sigma': (0.0, 0.5),    # Volatility of volatility
}

# Number of parameters to sample via LHS
n_params = len(param_ranges)

# Create a Latin Hypercube sampler with a random seed
# We set `seed=None` to ensure different samples each time
sampler = qmc.LatinHypercube(d=n_params, seed=None)

# Generate LHS samples
lhs_sample = sampler.random(n=N)

# Scale the samples to the parameter ranges
param_values = {}
for i, key in enumerate(param_ranges.keys()):
    low, high = param_ranges[key]
    param_values[key] = qmc.scale(lhs_sample[:, [i]], low, high).flatten()

# Combine all parameters into a DataFrame
data = pd.DataFrame()

# Add the LHS-sampled parameters to the DataFrame
for key in param_ranges.keys():
    data[key] = param_values[key]

# Initialize an empty DataFrame to hold all results
final_df = pd.DataFrame()

for index, row in data.iterrows():
    tau = row['tau']
    r = row['r']
    rho = row['rho']
    kappa = row['kappa']
    theta = row['theta']
    sigma = row['sigma']

    # Call the fdm function
    option_prices, moneyness, V_0 = fdm(time_to_maturity=tau, risk_free_rate=r, 
                                        reversion_speed=kappa, long_average_variance=theta, 
                                        vol_of_vol=sigma, correlation=rho)

    # Create a DataFrame for this parameter set
    df = pd.DataFrame({
        'moneyness': moneyness,
        'initial_variance': V_0,
        'tau': tau,
        'r': r,
        'rho': rho,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'option_price': option_prices
    })

    # Append the data from this iteration to the final DataFrame
    final_df = pd.concat([final_df, df], ignore_index=True)

    # Optional progress update
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} parameter sets")

# Write the final DataFrame to CSV once after the loop
output_filename = 'parameter_option_prices.csv'
final_df.to_csv(output_filename, mode='w', header=True, index=False)

end_time = time.time()
duration = end_time - start_time
print(f"Script ran for {duration:.2f} seconds")
print(f"Parameters and corresponding option prices saved to '{output_filename}'")
