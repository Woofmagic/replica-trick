import tensorflow as tf

from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model

from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LEARNING_RATE
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4
from statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5

def full_cross_section_dnn_model():
    
    try:

        # (1): Initialize the Network with Uniform Random Sampling: [-0.1, -0.1]:
        initializer = tf.keras.initializers.RandomUniform(
            minval = -1.0,
            maxval = 1.0,
            seed = None)

        # (2) Make the TF Input Layer:
        input_x_value = Input(shape = (1, ), name = 'input_layer')
    
        # (3): Define the Model Architecture:
        x1 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_1, activation = "relu", kernel_initializer = initializer)(input_x_value)
        x2 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_2, activation = "tanh", kernel_initializer = initializer)(x1)
        x3 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_3, activation = "relu", kernel_initializer = initializer)(x2)
        x4 = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_4, activation = "relu", kernel_initializer = initializer)(x3)
        output_y_value = Dense(_HYPERPARAMETER_NUMBER_OF_NEURONS_LAYER_5, activation = "linear", kernel_initializer = initializer, name = 'output_y_value')(x4)

        # (4): Define the model as as Keras Model:
        function_predictor_model = Model(inputs = input_x_value, outputs = output_y_value, name = "basic_function_predictor")

        print(function_predictor_model.summary())

        # (5): Compile the model with a fixed learning rate using Adam and a Loss of MSE:
        function_predictor_model.compile(
            optimizer = tf.keras.optimizers.Adam(_HYPERPARAMETER_LEARNING_RATE),
            loss = tf.keras.losses.MeanSquaredError())   

        # (6): Return the model:
        return function_predictor_model

    except Exception as ERROR:
        print(f"> Error in running DNN model:\n{ERROR}")

        return 0.