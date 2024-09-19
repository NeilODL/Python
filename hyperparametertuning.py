import pandas as pd
from sklearn.model_selection import train_test_split
from buildmodel import build_model  # Assuming this exists in your environment
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

class HestonHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=10, step=1)
        nodes = hp.Int('nodes', min_value=50, max_value=300, step=50)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            nodes, 
            input_dim=self.input_shape,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        
        for i in range(hidden_layers - 1):
            nodes_ = max(int(nodes / (i + 2)), 10)
            model.add(tf.keras.layers.Dense(
                nodes_, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ))
            model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        
        return model

def perform_hyperparameter_tuning(input_file='Fourier_Prices.csv'):
    # Load the standardized dataset
    df = pd.read_csv(input_file)
    print(f"Loaded filtered data from '{input_file}' with shape {df.shape}")
    
    # Define feature columns and target column
    feature_columns = ['moneyness', 'initial_variance', 'tau', 'r', 'rho', 'kappa', 'theta', 'sigma']
    target_column = 'option_price'
    
    # Separate features and target
    X = df[feature_columns].values
    y = df[target_column].values
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=123
    )
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Define the hypermodel
    hypermodel = HestonHyperModel(input_shape=X_train.shape[1])
    
    # Initialize the tuner
    tuner = RandomSearch(
        hypermodel,
        objective='val_mean_squared_error',
        max_trials=10,  # You can increase this based on computational resources
        executions_per_trial=1,
        directory='heston_tuning',  # Local directory to store temporary results
        project_name='heston_model'  # Name of the tuning project
    )
    
    # Define Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Run the hyperparameter search
    tuner.search(
        X_train, y_train,
        epochs=100,  # You can reduce or increase this as per your need
        batch_size=256,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\nThe hyperparameter search is complete. Here are the optimal values:\n")
    print(f"Optimal number of hidden layers: {best_hps.get('hidden_layers')}")
    print(f"Optimal number of nodes: {best_hps.get('nodes')}")
    print(f"Optimal dropout rate: {best_hps.get('dropout_rate')}")
    print(f"Optimal L2 regularization factor: {best_hps.get('l2_reg')}")
    
    return best_hps

if __name__ == "__main__":
    perform_hyperparameter_tuning()
