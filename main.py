import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model

from app.utilities.plotting.plot_customizer import PlotCustomizer

from app.extractions.network_builder import NetworkBuilder, WeightsBiasesCallback

# from app.utilities.plotting.plot_data import construct_plot

# from app.utilities.mathematics.sympy_function_generator import sympy_lambdify_expression
# from app.utilities.mathematics.sympy_function_generator import sympy_generate_random_function

# from app.utilities.mathematics.function_generator import generate_random_function

from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_FACTOR
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_PATIENCE
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_BATCH_SIZE
from app.statics.model_architecture.model_hyperparameters import _DNN_VERBOSE_SETTING

from app.data.experimental.experiment import conduct_experiment

def run():
    print(f"> Now running...")

    training_x_data_7, training_y_data_7 = conduct_experiment()

    # tensorflow_network_7 = NetworkBuilder().build_network()

    initializer = tf.keras.initializers.RandomUniform(
            minval = -1.0,
            maxval = 1.0,
            seed = None)


    input_x_value = Input(shape = (1, ), name = 'input_layer')
    
    # (3): Define the Model Architecture:
    x1 = Dense(10, activation = "relu", kernel_initializer = initializer)(input_x_value)
    x2 = Dense(10, activation = "relu", kernel_initializer = initializer)(x1)
    x3 = Dense(10, activation = "relu", kernel_initializer = initializer)(x2)
    x4 = Dense(10, activation = "relu", kernel_initializer = initializer)(x3)
    output_y_value = Dense(1, activation = "relu", kernel_initializer = initializer, name = 'output_y_value')(x4)

    # (4): Define the model as as Keras Model:
    tensorflow_network_7 = Model(inputs = input_x_value, outputs = output_y_value, name = "basic_function_predictor")
    
    # tensorflow_network_7.compile(optimizer='adam',
    #           loss = tf.keras.losses.MeanSquaredError(),
    #           metrics = ['accuracy'])
    
    tensorflow_network_7.compile(
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = 0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-07,
                amsgrad = False,
                weight_decay = None,
                clipnorm = None,
                clipvalue = None,
                global_clipnorm = None,
                use_ema = False,
                ema_momentum = 0.99,
                ema_overwrite_frequency = None,
                name='Optimizer_Adam',
            ),
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.MeanSquaredError()],
            loss_weights = None,
            weighted_metrics = None,
            run_eagerly = None,
            steps_per_execution = None)

    # (2): Train the network:
    # collect_weights_and_biases_callback = WeightsBiasesCallback(tensorflow_network_7)

    history_of_training_7 = tensorflow_network_7.fit(
        training_x_data_7, 
        training_y_data_7, 
        epochs = 2000)
        # validation_split = 0.2, # https://www.tensorflow.org/guide/keras/training_with_built_in_methods#automatically_setting_apart_a_validation_holdout_set
        # callbacks = [collect_weights_and_biases_callback],
        # callbacks = [
        #     tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = _HYPERPARAMETER_LR_FACTOR, patience = _HYPERPARAMETER_LR_PATIENCE, mode = 'auto'),
        #     tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience = _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER)
        # ],
        # batch_size = _HYPERPARAMETER_BATCH_SIZE,
        # verbose = _DNN_VERBOSE_SETTING
        )

    # (3): Construct the loss plot:
    training_loss_data_7 = history_of_training_7.history['loss']
    model_predictions_7 = tensorflow_network_7.predict(training_x_data_7)
    
    # (1): Set up the Figure instance
    figure_instance = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance = figure_instance.add_subplot(1, 3, 1)
    
    # (3): Customize the Axes Object:
    plot_customization = PlotCustomizer(
        axis_instance,
        title = r"Neural Network Performance",
        xlabel = r"Styff",
        ylabel = r"$FUCK$")
    
    plot_customization.add_line_plot(
        x_data = training_loss_data_7,
        label = r'$\\text{Training Loss}$',
        color = "black")

    plot_customization.add_scatter_plot(
        x_data = training_x_data_7,
        y_data = training_y_data_7,
        label = r'$\\text{True Data}$',
        color = "blue")
    
    plot_customization.add_scatter_plot(
        x_data = training_x_data_7,
        y_data = model_predictions_7,
        label = r'$\\text{Model Predictions}$',
        color = "red")

    plt.legend()
    plt.show()

    # weights_biases_history = collect_weights_and_biases_callback.weights_biases_history

    # for key, values in weights_biases_history.items():
    #     network_axes[2].plot(values, label = r"${}$".format(key))
    #     network_axes[2].legend()


if __name__ == "__main__":
    run()