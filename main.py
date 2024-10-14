import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model

from app.utilities.plotting.plot_customizer import PlotCustomizer

from app.extractions.network_builder import NetworkBuilder, WeightsBiasesCallback

# Native Library | datetime
import datetime

# Native Library | os
import os

from app.utilities.directories.handling_directories import create_replica_directories
from app.utilities.directories.handling_directories import create_replica_model_directories
from app.utilities.directories.handling_directories import create_replica_plots_directories
from app.utilities.directories.handling_directories import find_replica_model_directories
from app.utilities.directories.handling_directories import find_replica_plots_directories

from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_FACTOR
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_LR_PATIENCE
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_EARLYSTOP_PATIENCE_INTEGER
from app.statics.model_architecture.model_hyperparameters import _HYPERPARAMETER_BATCH_SIZE
from app.statics.model_architecture.model_hyperparameters import _DNN_VERBOSE_SETTING

from app.data.experimental.experiment import conduct_experiment

from app.extractions.replica_method import run_replica_method

def run():

    kinematic_set_integer = 1
    number_of_replicas = 5

    # run_replica_method(kinematic_set_integer, number_of_replicas)

    training_x_data_7, training_y_data_7 = conduct_experiment()

    # (1): Begin iterating over the replicas:
    for replica_index in range(number_of_replicas):

        initializer = tf.keras.initializers.RandomUniform(
            minval = -1.0,
            maxval = 1.0,
            seed = None)

        input_x_value = Input(shape = (1, ), name = 'input_layer')
        
        # (3): Define the Model Architecture:
        x1 = Dense(256, activation = "tanh", kernel_initializer = initializer)(input_x_value)
        x2 = Dense(256, activation = "relu", kernel_initializer = initializer)(x1)
        x3 = Dense(256, activation = "tanh", kernel_initializer = initializer)(x2)
        x4 = Dense(256, activation = "relu", kernel_initializer = initializer)(x3)
        x5 = Dense(256, activation = "tanh", kernel_initializer = initializer)(x4)
        output_y_value = Dense(1, activation = "linear", kernel_initializer = initializer, name = 'output_y_value')(x5)

        # (4): Define the model as as Keras Model:
        tensorflow_network = Model(inputs = input_x_value, outputs = output_y_value, name = "basic_function_predictor")
        
        tensorflow_network.compile(optimizer='adam',
                loss = tf.keras.losses.MeanSquaredError(),
                metrics = [tf.keras.metrics.MeanSquaredError()])

        history_of_training_7 = tensorflow_network.fit(
            training_x_data_7, 
            training_y_data_7, 
            epochs = 2000)

        # (3): Construct the loss plot:
        training_loss_data_7 = history_of_training_7.history['loss']
        model_predictions_7 = tensorflow_network.predict(training_x_data_7)
        
        # (1): Set up the Figure instance
        figure_instance_nn_loss = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_nn_loss = figure_instance_nn_loss.add_subplot(1, 1, 1)
        
        plot_customization_nn_loss = PlotCustomizer(
            axis_instance_nn_loss,
            title = r"Neural Network Loss",
            xlabel = r"Epoch",
            ylabel = r"Loss")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, 2000, 1),
            y_data = training_loss_data_7,
            label = r'Training Loss',
            color = "black")
        
        # (1): Set up the Figure instance
        figure_instance_fitting = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance_fitting = figure_instance_fitting.add_subplot(1, 1, 1)
        
        plot_customization_data_comparison = PlotCustomizer(
            axis_instance_fitting,
            title = r"Fitting Procedure",
            xlabel = r"x",
            ylabel = r"f(x)")

        plot_customization_data_comparison.add_scatter_plot(
            x_data = training_x_data_7,
            y_data = training_y_data_7,
            label = r'Experimental Data',
            color = "red")
        
        plot_customization_data_comparison.add_scatter_plot(
            x_data = training_x_data_7,
            y_data = model_predictions_7,
            label = r'Model Predictions',
            color = "orange")
        
        figure_instance_nn_loss.savefig(f"loss_v{replica_index+1}_v2")
        figure_instance_fitting.savefig(f"fitting{replica_index+1}_v2")


if __name__ == "__main__":
    run()