import pandas as pd
import time 
start_time = time.time()

input_filename = 'parameter_option_prices.csv'
output_filename = 'filtered_option_prices.csv'

# Initialize the CSV file with the header
header = True  # To write the header only once

chunksize = 10000  # Adjust the chunksize as needed

total_filtered = 0

for chunk in pd.read_csv(input_filename, chunksize=chunksize):
    
   # Apply the filters
    filtered_chunk = chunk[(chunk['moneyness'] > 0.6) & (chunk['moneyness'] < 1.4) &
                           (chunk['initial_variance'] > 0.05) & (chunk['initial_variance'] < 0.5)]
    
    # Debugging: Print the number of filtered data points in this chunk
    print(f"Number of filtered data points in this chunk: {len(filtered_chunk)}")

    # Write the filtered data to the CSV file
    filtered_chunk.to_csv(output_filename, mode='a' if not header else 'w', header=header, index=False)

    # Set header to False after the first write
    header = False

    total_filtered += len(filtered_chunk)
    print(f"Total filtered data points so far: {total_filtered}")

# Final output
print(f"Filtered data saved to '{output_filename}'")
print(f"Total number of data points after filtering: {total_filtered}")
end_time = time.time()
duration = end_time - start_time
print(f"Script ran for {duration:.2f} seconds")