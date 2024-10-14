import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def initialize_linear_model():
    """
    """
    # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
    initializer = tf.keras.initializers.RandomUniform(
        minval = -1.0,
        maxval = 1.0,
        seed = None)

    # (2) Make the TF Input Layer:
    input_x_value = layers.Input(shape = (1, ), name = 'input_layer')

    linear_model = tf.keras.Sequential([
        input_x_value,
        layers.Dense(units=1)
    ])


    print(linear_model.summary())