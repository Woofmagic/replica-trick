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

    print(f"> Now running TensorFlow Version {tf.version.VERSION}")

    # (1): Nonsense foor now
    kinematic_set_integer = 1

    # (2): The number of replicas to use.
    number_of_replicas = 300

    # run_replica_method(kinematic_set_integer, number_of_replicas)

    training_x_data_7, training_y_data_7 = conduct_experiment()

    # (1): Begin iterating over the replicas:
    for replica_index in range(number_of_replicas):

        initializer = tf.keras.initializers.RandomUniform(
            minval = -10.0,
            maxval = 10.0,
            seed = None)

        input_x_value = Input(shape = (1, ), name = 'input_layer')
        
        # (3): Define the Model Architecture:
        x1 = Dense(256, activation = "sigmoid", kernel_initializer = initializer)(input_x_value)
        x2 = Dense(256, activation = "sigmoid", kernel_initializer = initializer)(x1)
        x3 = Dense(256, activation = "sigmoid", kernel_initializer = initializer)(x2)
        x4 = Dense(256, activation = "sigmoid", kernel_initializer = initializer)(x3)
        x5 = Dense(256, activation = "sigmoid", kernel_initializer = initializer)(x4)
        output_y_value = Dense(1, activation = "linear", kernel_initializer = initializer, name = 'output_y_value')(x5)

        # (4): Define the model as as Keras Model:
        tensorflow_network = Model(inputs = input_x_value, outputs = output_y_value, name = "basic_function_predictor")
        
        tensorflow_network.compile(optimizer='adam',
                loss = tf.keras.losses.MeanSquaredError(),
                metrics = [tf.keras.metrics.MeanSquaredError()])
        
        tensorflow_network.summary()

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training_7 = tensorflow_network.fit(
            training_x_data_7, 
            training_y_data_7, 
            epochs = 2000)

        # (3): Construct the loss plot:
        training_loss_data_7 = history_of_training_7.history['loss']
        model_predictions_7 = tensorflow_network.predict(training_x_data_7)

        tensorflow_network.save(f"replica_number_{replica_index + 1}_v4.keras")

        print(f"> Saved replica!" )
        print(f"> Replica #{replica_index + 1} finished running...")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

        print(f"> Replica job finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")
        
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
        
        figure_instance_nn_loss.savefig(f"loss_v{replica_index+1}_v4")
        figure_instance_fitting.savefig(f"fitting{replica_index+1}_v4")

    model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith("v4.keras")]
    models = [tf.keras.models.load_model(path) for path in model_paths]

    print(f"> Obtained {len(models)} models!")

    def predict_with_models(models, x_values):

        y_mean = np.zeros(len(x_values))
        y_min = np.full(len(x_values), fill_value = np.inf)
        y_max = np.full(len(x_values), fill_value = - np.inf)
        y_q1 = np.zeros(len(x_values))
        y_q3 = np.zeros(len(x_values))
        all_predictions = []

        for model in models:

            y_prediction = model.predict(x_values).flatten()
            all_predictions.append(y_prediction)
            y_min = np.minimum(y_min, y_prediction)
            y_max = np.maximum(y_max, y_prediction)
            y_mean += y_prediction / len(models)

        all_predictions = np.array(all_predictions)

        y_percentile_10 = np.percentile(all_predictions, 10, axis = 0)
        y_percentile_20 = np.percentile(all_predictions, 20, axis = 0)
        y_percentile_30 = np.percentile(all_predictions, 30, axis = 0)
        y_percentile_40 = np.percentile(all_predictions, 40, axis = 0)
        y_percentile_60 = np.percentile(all_predictions, 50, axis = 0)
        y_percentile_70 = np.percentile(all_predictions, 60, axis = 0)
        y_percentile_80 = np.percentile(all_predictions, 70, axis = 0)
        y_percentile_90 = np.percentile(all_predictions, 80, axis = 0)

        return y_mean, y_min, y_max, y_percentile_10, y_percentile_20, y_percentile_30, y_percentile_40, y_percentile_60, y_percentile_70, y_percentile_80, y_percentile_90

    y_mean, y_min, y_max, y_percentile_10, y_percentile_20, y_percentile_30, y_percentile_40, y_percentile_60, y_percentile_70, y_percentile_80, y_percentile_90 = predict_with_models(models, training_x_data_7)

    # (1): Set up the Figure instance
    figure_instance_predictions = plt.figure(figsize = (18, 6))

    # (2): Add an Axes Object:
    axis_instance_predictions = figure_instance_predictions.add_subplot(1, 1, 1)
    
    plot_customization_predictions = PlotCustomizer(
        axis_instance_predictions,
        title = r"$N = {}$".format(number_of_replicas),
        xlabel = r"$x$",
        ylabel = r"$f(x)$")
    
    plot_customization_predictions.add_line_plot(
        x_data = training_x_data_7,
        y_data = y_mean,
        label = r'Replica Average',
        color = "blue",
        linestyle = "-")
    
    plot_customization_predictions.add_fill_between_plot(
        x_data = training_x_data_7,
        lower_y_data = y_min,
        upper_y_data = y_max,
        label = r'Min/Max Bound',
        color = "lightgray",
        alpha = 0.2)
    
    plot_customization_predictions.add_fill_between_plot(
        x_data = training_x_data_7,
        lower_y_data = y_percentile_10,
        upper_y_data = y_percentile_90,
        label = r'10/90 Bound',
        color = "gray",
        alpha = 0.25)

    plot_customization_predictions.add_fill_between_plot(
        x_data = training_x_data_7,
        lower_y_data = y_percentile_20,
        upper_y_data = y_percentile_80,
        label = r'20/80 Bound',
        color = "gray",
        alpha = 0.3)
    
    plot_customization_predictions.add_fill_between_plot(
        x_data = training_x_data_7,
        lower_y_data = y_percentile_30,
        upper_y_data = y_percentile_70,
        label = r'30/70 Bound',
        color = "gray",
        alpha = 0.35)
    
    plot_customization_predictions.add_fill_between_plot(
        x_data = training_x_data_7,
        lower_y_data = y_percentile_40,
        upper_y_data = y_percentile_60,
        label = r'40/60 Bound',
        color = "gray",
        alpha = 0.4)
    
    plot_customization_predictions.add_scatter_plot(
            x_data = training_x_data_7,
            y_data = training_y_data_7,
            label = r'Experimental Data',
            color = "red",
            marker = 'o',
            markersize = 1.)
    
    figure_instance_predictions.savefig(f"replica_average_data_v4")



if __name__ == "__main__":
    run()