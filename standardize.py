# standardize_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def standardize_dataset(input_file='filtered_option_prices.csv', 
                        output_file='standardized_option_prices.csv'):
    """
    Standardizes the features and target variable in the dataset.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the standardized CSV file.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    print(f"Loaded data from '{input_file}' with shape {df.shape}")
    
    # Define feature columns and target column
    feature_columns = ['moneyness', 'initial_variance', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma']
    target_column = 'option_price'
    
    # Check if all required columns are present
    missing_features = set(feature_columns + [target_column]) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing columns in input data: {missing_features}")
    
    # Separate features and target
    X = df[feature_columns]
    y = df[[target_column]]  # Keep as DataFrame for scaler
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit and transform features
    X_scaled = scaler_X.fit_transform(X)
    print("Features have been standardized.")
    
    # Fit and transform target
    y_scaled = scaler_y.fit_transform(y)
    print("Target variable has been standardized.")
    
    # Combine scaled features and target into a DataFrame
    standardized_df = pd.DataFrame(X_scaled, columns=feature_columns)
    standardized_df[target_column] = y_scaled
    print(f"Standardized data has shape {standardized_df.shape}")
    
    # Save the standardized dataset to CSV
    standardized_df.to_csv(output_file, index=False)
    print(f"Standardized data saved to '{output_file}'")
    
    # Save the scaler objects for future inverse transformations
    joblib.dump(scaler_X, 'scaler_X.save')
    joblib.dump(scaler_y, 'scaler_y.save')
    print("Scaler objects saved as 'scaler_X.save' and 'scaler_y.save'")

if __name__ == "__main__":
    standardize_dataset()
