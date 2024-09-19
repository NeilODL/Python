import pandas as pd
from sklearn.model_selection import train_test_split
from buildmodel import build_model
import tensorflow as tf
import joblib
import json
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_neural_network(input_file='Fourier_Prices.csv',
                         model_save_path='trained_model.keras',
                         epochs=50,
                         batch_size=256,
                         hidden_layers=1,
                         nodes=300,
                         dropout_rate=0.1,
                         l2_reg=2.3553594488703456e-05,
                         test_size=0.20,
                         random_state=123):
    """
    Trains a neural network on the standardized dataset and saves predictions.

    Parameters:
    - input_file (str): Path to the standardized CSV file.
    - model_save_path (str): Path to save the trained model.
    - epochs (int): Number of training epochs.
    - batch_size (int): Size of training batches.
    - hidden_layers (int): Number of hidden layers in the model.
    - nodes (int): Number of nodes in the first hidden layer.
    - dropout_rate (float): Dropout rate for regularization.
    - l2_reg (float): L2 regularization factor.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed used by the random number generator.
    """
    # Load the standardized dataset
    df = pd.read_csv(input_file)
    print(f"Loaded standardized data from '{input_file}' with shape {df.shape}")
    
    # Define feature columns and target column
    feature_columns = ['moneyness', 'initial_variance', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma']
    target_column = 'option_price'
    
    # Separate features and target
    X = df[feature_columns].values
    y = df[target_column].values
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Build the neural network model
    model = build_model(
        inp_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        nodes=nodes,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )
    print("Neural network model has been built.")
    
    # Display the model architecture
    model.summary()
    
    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    print("Training completed.")
    
    # Save the trained model
    model.save(model_save_path)
    print(f"Trained model saved to '{model_save_path}'")
    
    # Save the training history for future analysis
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    print("Training history saved to 'training_history.json'")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Save actual and predicted values to a JSON file for future analysis
    predictions = {
        "actual_values": y_test.tolist(),
        "predicted_values": y_pred.flatten().tolist()  # Flatten to avoid nested arrays
    }
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)
    print("Predictions saved to 'predictions.json'")

if __name__ == "__main__":
    train_neural_network()
