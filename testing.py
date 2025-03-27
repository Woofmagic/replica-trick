import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
x = np.linspace(-1, 1, 100)
y_true = 0.65 * x - 0.18
y = y_true + np.random.normal(0, 0.05, size=x.shape)  # Adding noise

# Split into training and validation sets
x_train, x_val = x[:80], x[80:]
y_train, y_val = y[:80], y[80:]

# Build the model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(1,)), 
    layers.Dense(16, activation='relu'), 
    layers.Dense(1)  # Output layer (linear activation)
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mse', 
              metrics=['mae'])

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val), 
                    epochs=500, 
                    batch_size=10, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Evaluate model
val_loss, val_mae = model.evaluate(x_val, y_val)
print(f'Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}')

# Plot predictions
x_test = np.linspace(-1, 1, 100)
y_pred = model.predict(x_test)

plt.scatter(x, y, label="Data", color='blue', alpha=0.5)
plt.plot(x_test, y_pred, label="DNN Prediction", color='red')
plt.plot(x_test, 0.65*x_test - 0.18, label="True Function", color='green', linestyle="dashed")
plt.legend()
plt.show()
