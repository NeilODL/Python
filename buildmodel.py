# build_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers, regularizers

def build_model(inp_size, hidden_layers=2, nodes=100, dropout_rate=0.2, l2_reg=1e-4):
    """
    Builds a Sequential neural network model with regularization.

    Parameters:
    - inp_size (int): Number of input features.
    - hidden_layers (int): Number of hidden layers.
    - nodes (int): Number of nodes in the first hidden layer.
    - dropout_rate (float): Fraction of neurons to drop.
    - l2_reg (float): L2 regularization factor.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    # Initialize the Glorot uniform initializer
    glorot_uniform = initializers.GlorotUniform(seed=None)
    
    # Initialize the Sequential model
    model = Sequential()
    
    # Add the first hidden layer with input dimension
    model.add(Dense(
        nodes, 
        input_dim=inp_size,
        kernel_initializer=glorot_uniform,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    
    # Add Dropout
    model.add(Dropout(rate=dropout_rate))
    
    # Add additional hidden layers if specified
    for i in range(hidden_layers - 1):
        # Decrease the number of nodes in subsequent layers
        nodes_ = max(int(nodes / (i + 2)), 10)  # Ensure at least 10 nodes
        model.add(Dense(
            nodes_, 
            kernel_initializer=glorot_uniform,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(Dropout(rate=dropout_rate))
    
    # Add the output layer
    model.add(Dense(
        1, 
        kernel_initializer=initializers.RandomNormal(),
        activation='linear'  # Suitable for regression tasks
    ))
    
    # Compile the model with mean squared error loss and Adam optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['mean_squared_error']
    )
    
    return model

# Optional: Test the build_model function
if __name__ == "__main__":
    # Example: Build a model with 8 input features, 2 hidden layers, and 100 nodes
    model = build_model(inp_size=8, hidden_layers=2, nodes=100)
    model.summary()
