import json
import matplotlib.pyplot as plt
import numpy as np

# Load the training history from JSON
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Extract data for plotting
loss = history.get('loss', [])
val_loss = history.get('val_loss', [])

# Optional: Check if 'mean_absolute_error' exists for additional plotting
mae = history.get('mean_absolute_error', [])
val_mae = history.get('val_mean_absolute_error', [])

# Plot 1: Training and Validation Loss (MSE) vs. Epochs
epochs = range(1, len(loss) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'bo-', label='Training loss (MSE)')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss (MSE)')
plt.title('Training and Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Training and Validation MAE (if available)
if mae and val_mae:
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mae, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
    plt.title('Training and Validation Mean Absolute Error (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load actual and predicted values from predictions.json
with open('predictions.json', 'r') as f:
    predictions_data = json.load(f)

actual_values = np.array(predictions_data['actual_values'])
predicted_values = np.array(predictions_data['predicted_values'])

# Plot 3: Predicted vs. Actual (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predicted_values, alpha=0.5)
plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--')
plt.title('Predicted vs Actual Option Prices')
plt.xlabel('Actual Option Prices')
plt.ylabel('Predicted Option Prices')
plt.grid(True)
plt.show()

# Plot 4: Residuals Plot (Actual - Predicted)
residuals = actual_values - predicted_values
plt.figure(figsize=(10, 6))
plt.scatter(predicted_values, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.title('Residuals Plot (Actual - Predicted)')
plt.xlabel('Predicted Option Prices')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Plot 5: Histogram of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7, color='blue')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
