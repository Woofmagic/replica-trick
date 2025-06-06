# Native Library | datetime
import datetime

# Native Library | os
import os

# Native Library | sys
import sys

# Native Library | re
import re

from app.utilities.plotting.plot_customizer import PlotCustomizer

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

# External Library | NumPy
import numpy as np

# External Library | Matplotlib
import matplotlib.pyplot as plt
# rcParams stuff....
plt.rcParams.update(plt.rcParamsDefault)

# External Library | TensorFlow
import tensorflow as tf

# External Library | TensorFlow | Keras | Layers | Input & Dense
from tensorflow.keras.layers import Input, Dense

# External Library | TensorFlow | Keras | Models | Model
from tensorflow.keras.models import Model

from tensorflow import keras

# External Library | PySR
from pysr import PySRRegressor

# External Library | Pandas
import pandas as pd

# External Library | SymPy
import sympy as sp

from matplotlib import rcParams

rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.direction': 'in',
    'xtick.major.size': 5,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 2.5,
    'xtick.minor.width': 0.5,
    'xtick.minor.visible': True,
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.major.size': 5,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 2.5,
    'ytick.minor.width': 0.5,
    'ytick.minor.visible': True,
    'ytick.right': True,
})

SETTING_VERBOSE = True
SETTING_DEBUG = True
LEARNING_RATE = 0.005
BATCH_SIZE_LOCAL_FITS = 32
BATCH_SIZE_GLOBAL_FITS = 10
EARLY_STOP_PATIENCE = 20
LEARNING_RATE_PATIENCE = 20
MODIFY_LR_FACTOR = 0.9
SETTING_DNN_TRAINING_VERBOSE = 1

NUMBER_OF_REPLICAS = 1
EPOCHS = 1000

_SEARCH_SPACE_BINARY_OPERATORS = [
    "+", "-", "*", "/"
]

_SEARCH_SPACE_UNARY_OPERATORS = [
    
]

_SEARCH_SPACE_MAXIUMUM_COMPLEXITY = 25

_SEARCH_SPACE_MAXIMUM_DEPTH = None

py_regressor_models = PySRRegressor(

    # === SEARCH SPACE ===

    # (1): Binary operators:
    binary_operators = _SEARCH_SPACE_BINARY_OPERATORS,

    # (2): Unary operators:
    unary_operators = _SEARCH_SPACE_UNARY_OPERATORS,

    # (3): Maximum complexity of chosen equation:
    maxsize = _SEARCH_SPACE_MAXIUMUM_COMPLEXITY,

    # (4): Maximum depth of a chosen equation:
    maxdepth = _SEARCH_SPACE_MAXIMUM_DEPTH,

    # === SEARCH SIZE ===

    # (1): Number of iterations for the algorithm:
    niterations = 1000,

    # (2): The number of "populations" running:
    populations = 500,

    # (3): The size of each population:
    population_size = 50,

    # (4): Whatever this means:
    ncycles_per_iteration = 550,

    # === OBJECTIVE ===

    # (1): Option 1: Specify *Julia* code to compute elementwise loss:
    elementwise_loss = "loss(prediction, target) = (prediction - target)^2",

    # (2): Option 2: Code your own *Julia* loss:
    loss_function = None,

    # (3): Choose the "metric" to select the final function --- can be 'accuracy,' 'best', or 'score':
    model_selection = 'best',

    # (4): How much to penalize a given function if dim-analysis doesn't work:
    dimensional_constraint_penalty = 1000.0,

    # (5): Enable or disable a search for dimensionless constants:
    dimensionless_constants_only = False,

    # === COMPLEXITY ===

    # (1): Multiplicative factor that penalizes a complex function:
    parsimony = 0.03,

    # (2): A complicated dictionary governing how complex a given operation can be:
    constraints = None,

    # (3): Another dictionary that enforces the number of times an operator may be nested:
    nested_constraints = None,

    # (4): Another dictionary that limits the complexity per operator:
    complexity_of_operators = None)

def get_next_version(base_path: str) -> str:
    """
    Scans the base_path (e.g., 'science/analysis/' or 'science/data/') to find the highest version_x directory,
    then returns the next version path.
    """

    # (X): 
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # (X): 
    version_pattern = re.compile(r'version_(\d+)')
    
    existing_versions = []
    
    for entry in os.listdir(base_path):
        match = version_pattern.match(entry)

        if match:
            existing_versions.append(int(match.group(1)))
    
    next_version = max(existing_versions, default = -1) + 1
    
    return next_version

def generate_replica_data(
        pandas_dataframe: pd.DataFrame,
        mean_value_column_name: str,
        stddev_column_name: str,
        new_column_name: str):
    """
        ## Description:
        Generates a replica dataset by sampling the mean within its standard deviation.
    """
    pseudodata_dataframe = pandas_dataframe.copy()

    # (): Overwrites what's in the "stddev" column of the original DF:
    pseudodata_dataframe[stddev_column_name] = pandas_dataframe[stddev_column_name]

    # Generate normally distributed F values
    replica_cross_section_sample = np.random.normal(
        loc = pandas_dataframe[mean_value_column_name], 
        scale = pandas_dataframe[stddev_column_name])

    # (): Write a new column (mean values) on the pseudodata dataframe:
    pseudodata_dataframe[new_column_name] = replica_cross_section_sample

    return pseudodata_dataframe

def split_data(x_data, y_data, y_error_data, split_percentage = 0.1):
    """Splits data into training and testing sets based on a random selection of indices."""
    test_indices = np.random.choice(
        x_data.index,
        size = int(len(y_data) * split_percentage),
        replace = False)

    train_X = x_data.loc[~x_data.index.isin(test_indices)]
    test_X = x_data.loc[test_indices]

    train_y = y_data.loc[~y_data.index.isin(test_indices)]
    test_y = y_data.loc[test_indices]

    train_yerr = y_error_data.loc[~y_error_data.index.isin(test_indices)]
    test_yerr = y_error_data.loc[test_indices]

    return train_X, test_X, train_y, test_y, train_yerr, test_yerr

def run():

    _PATH_SCIENCE_ANALYSIS = 'app/science/analysis/'

    # Get next version directories
    _version_number = get_next_version(_PATH_SCIENCE_ANALYSIS)

    if SETTING_VERBOSE:
        print(f"> Determined next analysis directory: {_version_number}")

    DATA_FILE_NAME = f"E{_version_number}_raw_data.csv"

    if SETTING_VERBOSE:
        print(f"> Computed the name of the raw data file to be: {DATA_FILE_NAME}")

    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/data/raw')
    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/data/replicas')

    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/pseudodata')
    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/losses')
    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/fits')
    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/performance')

    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/sr_analysis/replica_average_sr')
    os.makedirs(f'{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/sr_analysis/replica_median_sr')

    try:
        # tf.config.set_visible_devices([],'GPU')
        tensorflow_found_devices = tf.config.list_physical_devices()

        if len(tf.config.list_physical_devices()) != 0:
            for device in tensorflow_found_devices:
                if SETTING_VERBOSE:
                    print(f"> TensorFlow detected device: {device}")

        else:
            if SETTING_VERBOSE:
                print("> TensorFlow didn't find CPUs or GPUs...")

    except Exception as error:
        print(f"> TensorFlow could not find devices due to error:\n> {error}")

    if SETTING_VERBOSE:
        print(f"> Now running TensorFlow Version {tf.version.VERSION}")

    # (1): Nonsense for now
    kinematic_set_integer = 1
    
    # run_replica_method(kinematic_set_integer, number_of_replicas)

    # (X): Conduct an experiment:
    experimental_x_data, experimental_y_data, experimental_y_error_data = conduct_experiment(_version_number)

    # (1): Begin iterating over the replicas:
    for replica_index in range(NUMBER_OF_REPLICAS):
        
        # (X): Rely on Pandas to correctly read the just-generated .csv file:
        this_replica_data_set = pd.read_csv(DATA_FILE_NAME)

        # (X): For every replica, we generate associated replica data using `generate_replica_data`:
        # - this works even for n > 1 (right now), but ONLY for when there is *one* output y value!
        pseudodata_dataframe = generate_replica_data(
            pandas_dataframe = this_replica_data_set,
            mean_value_column_name = 'y',
            stddev_column_name = 'y_error',
            new_column_name = 'y_pseudodata')
        
        # (X): We also store the pseudodata/replica data for reproducability purposes:
        pseudodata_dataframe.to_csv(
            path_or_buf = f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/data/raw/pseudodata_replica_{replica_index+1}_data_v{_version_number}.csv",
            index_label = None)

        # (X): Partition the replica data into training and testing data:
        independent_variable_dataframe_columns = [column for column in pseudodata_dataframe.columns if column.startswith("x_")]

        training_x_data, testing_x_data, training_y_data, testing_y_data, training_y_error, testing_y_error = split_data(
            x_data = pseudodata_dataframe[independent_variable_dataframe_columns],
            y_data = pseudodata_dataframe['y'],
            y_error_data = pseudodata_dataframe['y_pseudodata'],
            split_percentage = 0.2)

        # (X): TEMPORARY!
        function_input_dimension = 2

        figure_instance_pseudodata = plt.figure(figsize = (10, 5.5))
        
        if function_input_dimension == 1:

            # (2): Add an Axes Object:
            axis_instance_pseudodata = figure_instance_pseudodata.add_subplot(1, 1, 1)
            
            plot_customizer_psueodata = PlotCustomizer(
                axis_instance_pseudodata,
                title = rf"Pseudodata Generation for Replica ${replica_index + 1}$",
                xlabel = r"$x$",
                ylabel = r"$y\left(x\right)$")
            
            plot_customizer_psueodata.add_errorbar_plot(
                x_data = this_replica_data_set['x_1'],
                y_data = this_replica_data_set['y'],
                x_errorbars = np.zeros(this_replica_data_set['x_1'].shape),
                y_errorbars = this_replica_data_set['y_error'],
                label = 'Raw Data',
                color = "black",)
            
            plot_customizer_psueodata.add_errorbar_plot(
                x_data = pseudodata_dataframe['x_1'],
                y_data = pseudodata_dataframe['y_pseudodata'],
                x_errorbars = np.zeros(pseudodata_dataframe['y_pseudodata'].shape),
                y_errorbars = np.zeros(pseudodata_dataframe['y_pseudodata'].shape),
                label = 'Generated Pseudodata',
                color = "orange",)
        
        elif function_input_dimension == 2:

            # (2): Add an Axes Object:
            axis_instance = figure_instance_pseudodata.add_subplot(1, 1, 1, projection = "3d")

            # (3): Customize the Axes Object:
            plot_customization = PlotCustomizer(
                axis_instance,
                title = rf"Pseudodata Generation for Replica ${replica_index + 1}$",
                xlabel = r"$x$",
                ylabel = r"$y$",
                zlabel = r"$f(x, y)$",
                grid = True)

            # (4): Add data to the Axes Object:
            plot_customization.add_3d_scatter_plot(
                x_data = this_replica_data_set['x_1'],
                y_data = this_replica_data_set['x_2'],
                z_data = this_replica_data_set['y'],
                label = "Raw Data",
                color = "black",
                alpha = 0.4)
            
            plot_customization.add_3d_scatter_plot(
                x_data = pseudodata_dataframe['x_1'],
                y_data = pseudodata_dataframe['x_2'],
                z_data = pseudodata_dataframe['y_pseudodata'],
                label = "Generated Pseudodata",
                color = "orange",
                alpha = 0.8)
            
        else:
            raise NotImplementedError("> Functions of n > 2 independent variables are currently unavailable for analysis!")
        
        figure_instance_pseudodata.savefig(
            fname = f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/pseudodata/generated_pseudodata_replica_{replica_index + 1}_v{_version_number}.png")

        input_x_value = Input(
            shape = (function_input_dimension, ),
            name = 'input_layer')

        # (3): Define the Model Architecture:
        x1 = Dense(32, activation = "relu")(input_x_value)
        x2 = Dense(16, activation = "relu")(x1)
        output_y_value = Dense(1, activation = "linear", name = 'output_y_value')(x2)

        # (4): Define the model as as Keras Model:
        tensorflow_network = Model(
            inputs = input_x_value,
            outputs = output_y_value,
            name = "basic_function_predictor")
        
        tensorflow_network.compile(
            optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE),
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
                ])

        if SETTING_VERBOSE:
            tensorflow_network.summary()

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        if SETTING_VERBOSE:
            print(f"> Replica #{replica_index + 1} now running...")

        history_of_training = tensorflow_network.fit(
            training_x_data,
            training_y_data,
            validation_data = (testing_x_data, testing_y_data),
            epochs = EPOCHS,
            callbacks = [
                # tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = modify_LR_factor, patience = LEARNING_RATE_PATIENCE, mode = 'auto'),
                # tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = LEARNING_RATE_PATIENCE)
            ],
            batch_size = BATCH_SIZE_LOCAL_FITS,
            verbose = SETTING_DNN_TRAINING_VERBOSE)

        # (X): Extract the training loss info from the `history` dictionary with key "loss":
        training_loss_data = history_of_training.history['loss']

        # (X): Validation loss information is also available in `history` dictionary because we used validation data:
        validation_loss_data = history_of_training.history['val_loss']

        # (X): Use TF's `.evaluate()` to assess losses:
        validaton_loss, validation_mae = tensorflow_network.evaluate(testing_x_data, testing_y_data)
        
        if SETTING_VERBOSE:
            print(f'Validation Loss: {validaton_loss:.4f}, Validation MAE: {validation_mae:.4f}')

        # (X):
        # model_predictions = np.array(tensorflow_network.predict(training_x_data))
        model_predictions = np.array(tensorflow_network.predict(pseudodata_dataframe[independent_variable_dataframe_columns]))

        try:
            tensorflow_network.save(f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/data/replicas/replica_number_{replica_index + 1}_v{_version_number}.keras")
            
            if SETTING_VERBOSE:
                print("> Saved replica!")

        except Exception as error:
            print(f"> Error saving replica:\n> {error}!")
            sys.exit(0)

        if SETTING_VERBOSE:
            print(f"> Replica #{replica_index + 1} finished running...")
    
        end_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)

        if SETTING_VERBOSE:
            print(f"> Replica job finished in {end_time_in_milliseconds - start_time_in_milliseconds}ms.")
        
        # (1): Set up the Figure instance
        figure_instance_nn_loss = plt.figure(figsize = (10, 5.5))

        # (2): Add an Axes Object:
        axis_instance_nn_loss = figure_instance_nn_loss.add_subplot(1, 1, 1)
        
        plot_customization_nn_loss = PlotCustomizer(
            axis_instance_nn_loss,
            title = "Neural Network Loss per Epoch",
            xlabel = "Epoch",
            ylabel = "Loss (MSE)")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = np.array([np.max(training_loss_data) for number in training_loss_data]),
            color = "red",
            linestyle = ':')
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = training_loss_data,
            label = 'Training Loss',
            color = "orange")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = validation_loss_data,
            label = 'Validation Loss',
            color = "pink")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
            y_data = np.zeros(shape = EPOCHS),
            color = "limegreen",
            linestyle = ':')
        
        figure_instance_nn_loss.savefig(f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/losses/loss_v{replica_index+1}_v{_version_number}.png")

        # (1): Set up the Figure instance
        figure_instance_fitting = plt.figure(figsize = (10, 5.5))
        
        if function_input_dimension == 1:

            # (2): Add an Axes Object:
            axis_instance_fitting = figure_instance_fitting.add_subplot(1, 1, 1)
        
            # (3): Customize the Axes Object:
            plot_customization_data_comparison = PlotCustomizer(
                axis_instance_fitting,
                title = "Fitting Procedure",
                xlabel = r"$x$",
                ylabel = r"$f(x)$")
            
            # (4): Add data to the Axes Object:
            plot_customization_data_comparison.add_errorbar_plot(
                x_data = experimental_x_data,
                y_data = experimental_y_data,
                x_errorbars = np.array([0.]),
                y_errorbars = experimental_y_error_data,
                label = 'Experimental Data',
                color = 'red')
            
            plot_customization_data_comparison.add_errorbar_plot(
                x_data = pseudodata_dataframe['x_1'],
                y_data = pseudodata_dataframe['y_pseudodata'],
                x_errorbars = np.zeros(pseudodata_dataframe['y_pseudodata'].shape),
                y_errorbars = np.zeros(pseudodata_dataframe['y_pseudodata'].shape),
                label = 'Replica Pseudodata',
                color = "orange")
            
            # (5): Add data to the Axes Object:
            plot_customization_data_comparison.add_scatter_plot(
                x_data = training_x_data,
                y_data = model_predictions,
                label = 'Model Predictions',
                color = "blue",
                markersize = 4.)
            
        elif function_input_dimension == 2:

            # (2): Add an Axes Object:
            axis_instance_fitting = figure_instance_fitting.add_subplot(1, 1, 1, projection = "3d")

            # (3): Customize the Axes Object:
            plot_customization_data_comparison = PlotCustomizer(
                axis_instance_fitting,
                title = "Fitting Procedure",
                xlabel = r"$x_1$",
                ylabel = r"$x_0, :2$",
                zlabel = r"$f(x_1, x_2$)",
                grid = True)

            # (4): Add data to the Axes Object:
            plot_customization_data_comparison.add_3d_error_scatter_plot(
                x_data = experimental_x_data[:, 0],
                y_data = experimental_x_data[:, 1],
                z_data = experimental_y_data,
                z_errors = experimental_y_error_data,
                label = "Experimental Data",
                color = "red",
                alpha = 0.8)
            
            plot_customization_data_comparison.add_3d_scatter_plot(
                x_data = experimental_x_data[:, 0],
                y_data = experimental_x_data[:, 1],
                z_data = model_predictions.flatten(),
                label = 'Model Predictions',
                color = "blue",
                alpha = 0.5)
            
        else:
            raise NotImplementedError("> Functions of n > 2 independent variables are currently unavailable for analysis!")
        
        figure_instance_fitting.savefig(f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/fits/fitting_replica_{replica_index+1}_v{_version_number}.png")

    model_paths = [os.path.join(os.getcwd(), f"app/science/analysis/version_{_version_number}/data/replicas/{file}") for file in os.listdir(f"app/science/analysis/version_{_version_number}/data/replicas") if file.endswith(".keras")]
    
    models = [tf.keras.models.load_model(path) for path in model_paths]

    print(f"> Obtained {len(models)} models!")

    raw_data = pd.read_csv(f"E{_version_number}_raw_data.csv")
    training_x_data = raw_data[independent_variable_dataframe_columns]
    training_y_data = raw_data['y']
    y_error_data = raw_data['y_error']

    def predict_with_models(models, x_values):
        x_values = np.array(x_values)

        if x_values.ndim == 1:
            x_values = x_values.reshape(-1, 1)
        
        all_predictions = np.array([
            model.predict(x_values).flatten() for model in models
            ])

        y_mean = np.mean(all_predictions, axis = 0)
        y_min = np.min(all_predictions, axis = 0)
        y_max = np.max(all_predictions, axis = 0)
        y_q1 = np.percentile(all_predictions, 25, axis = 0)
        y_q3 = np.percentile(all_predictions, 75, axis = 0)

        y_percentile_10 = np.percentile(all_predictions, 10, axis = 0)
        y_percentile_20 = np.percentile(all_predictions, 20, axis = 0)
        y_percentile_30 = np.percentile(all_predictions, 30, axis = 0)
        y_percentile_40 = np.percentile(all_predictions, 40, axis = 0)
        y_median = np.percentile(all_predictions, 50, axis = 0)
        y_percentile_60 = np.percentile(all_predictions, 50, axis = 0)
        y_percentile_70 = np.percentile(all_predictions, 60, axis = 0)
        y_percentile_80 = np.percentile(all_predictions, 70, axis = 0)
        y_percentile_90 = np.percentile(all_predictions, 80, axis = 0)

        return y_mean, y_min, y_max, y_q1, y_q3, y_percentile_10, y_percentile_20, y_percentile_30, y_percentile_40, y_median, y_percentile_60, y_percentile_70, y_percentile_80, y_percentile_90

    y_mean, y_min, y_max, y_q1, y_q3, y_percentile_10, y_percentile_20, y_percentile_30, y_percentile_40, y_median, y_percentile_60, y_percentile_70, y_percentile_80, y_percentile_90 = predict_with_models(models, training_x_data)

    # (1): Set up the Figure instance
    figure_instance_predictions = plt.figure(figsize = (10, 5.5))

    if function_input_dimension == 1:

        # (2): Add an Axes Object:
        axis_instance_predictions = figure_instance_predictions.add_subplot(1, 1, 1)
        
        plot_customization_predictions = PlotCustomizer(
            axis_instance_predictions,
            title = rf"Replica Average for $N = {NUMBER_OF_REPLICAS}$",
            xlabel = r"$x$",
            ylabel = r"$f(x)$")
        
        plot_customization_predictions.add_line_plot(
            x_data = training_x_data,
            y_data = y_mean,
            label = r'Replica Average',
            color = "blue",
            linestyle = "-")
        
        plot_customization_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_min,
            upper_y_data = y_max,
            label = r'Min/Max Bound',
            color = "lightgray",
            alpha = 0.2)
        
        plot_customization_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_percentile_10,
            upper_y_data = y_percentile_90,
            label = r'10/90 Bound',
            color = "gray",
            alpha = 0.25)

        plot_customization_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_percentile_20,
            upper_y_data = y_percentile_80,
            label = r'20/80 Bound',
            color = "gray",
            alpha = 0.3)
        
        plot_customization_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_percentile_30,
            upper_y_data = y_percentile_70,
            label = r'30/70 Bound',
            color = "gray",
            alpha = 0.35)
        
        plot_customization_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_percentile_40,
            upper_y_data = y_percentile_60,
            label = r'40/60 Bound',
            color = "gray",
            alpha = 0.4)
        
        plot_customization_predictions.add_scatter_plot(
                x_data = training_x_data,
                y_data = training_y_data,
                label = r'Experimental Data',
                color = "red",
                marker = 'o',
                markersize = 1.)
        
    elif function_input_dimension == 2:

        # (2): Add an Axes Object:
        axis_instance_predictions = figure_instance_predictions.add_subplot(1, 1, 1, projection = '3d')
        
        plot_customization_predictions = PlotCustomizer(
            axis_instance_predictions,
            title = rf"Replica Average for $N = {NUMBER_OF_REPLICAS}$",
            xlabel = r"$x_1$",
            ylabel = r"$x_2$",
            zlabel = r"$f(x_1, x_2)$",
            grid = True)
        
        training_x_grid_data, training_y_grid_data = np.meshgrid(
            np.sort(training_x_data['x_1'].unique()),
            np.sort(training_x_data['x_2'].unique()),
            indexing = "ij"
        )
        
        plot_customization_predictions.add_surface_plot(
            x_data = training_x_grid_data,
            y_data = training_y_grid_data,
            z_data = y_mean.reshape(
                len(training_x_data['x_1'].unique()),
                len(training_x_data['x_2'].unique())
            ),
            label = r'Replica Average',
            colormap = "inferno",
            alpha = 0.7)
        
        plot_customization_predictions.add_surface_plot(
            x_data = training_x_grid_data,
            y_data = training_y_grid_data,
            z_data = y_percentile_10.reshape(
                len(training_x_data['x_1'].unique()),
                len(training_x_data['x_2'].unique())
            ),
            label = r'10th Percentile Bound',
            colormap = "gray",
            alpha = 0.3)
        
        plot_customization_predictions.add_surface_plot(
            x_data = training_x_grid_data,
            y_data = training_y_grid_data,
            z_data = y_percentile_90.reshape(
                len(training_x_data['x_1'].unique()),
                len(training_x_data['x_2'].unique())
            ),
            label = r'90th Percentile Bound',
            colormap = "gray",
            alpha = 0.3)
        
        plot_customization_predictions.add_3d_error_scatter_plot(
                x_data = experimental_x_data[:, 0],
                y_data = experimental_x_data[:, 1],
                z_data = experimental_y_data,
                z_errors = experimental_y_error_data,
                color = "red",
                alpha = 0.8)
        
    figure_instance_predictions.savefig(
        fname = f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/performance/replica_average_data_v{_version_number}")
    plt.close()

    # (X): Run the PySR fitting procedure:
    py_regressor_models.fit(
        X = pd.DataFrame(training_x_data),
        y = y_mean)
    
    print(f"> Best fit model for this: {py_regressor_models.latex()}")

    # (X): Sort the equations according to their LOSS in the PySR algorithm:
    sorted_equations = sorted(
        py_regressor_models.equations_.itertuples(),
        key = lambda eq: eq.loss,
        reverse = True)

    # (X): We assign a list of colors for each equation in our output:
    colors = plt.cm.jet(np.linspace(0, 1, len(sorted_equations)))

    figure_pysr_predictions = plt.figure(figsize = (18, 6))

    if function_input_dimension == 1:

        axis_instance_pysr_predictions = figure_pysr_predictions.add_subplot(1, 1, 1)

        plot_customization_pysr_predictions = PlotCustomizer(
            axis_instance_pysr_predictions,
            title = fr"Replica Method Predictions for $N = {NUMBER_OF_REPLICAS}$",
            xlabel = r"$x$",
            ylabel = r"$f(x)$",)
        
        plot_customization_pysr_predictions.add_line_plot(
            x_data = training_x_data,
            y_data = y_mean,
            label = "Replica Average",
            color = "blue",
            linestyle = '-')
        
        plot_customization_pysr_predictions.add_fill_between_plot(
            x_data = training_x_data,
            lower_y_data = y_q1,
            upper_y_data = y_q3,
            label = "IQR",
            color = "lightgray",
            alpha = 0.34,)
        
        plot_customization_pysr_predictions.add_errorbar_plot(
            x_data = training_x_data,
            y_data = training_y_data,
            x_errorbars = np.zeros(y_error_data.shape),
            y_errorbars = y_error_data,
            label = "Experimental Data",
            color = "red",
            marker = 'o',)
        
        for index, (equation, color) in enumerate(zip(sorted_equations, colors)):

            # (X): Extract the equation's complexity with `.complexity`:
            equation_complexity = equation.complexity

            # (X): Extract the loss during fitting with `.loss`:
            equation_loss = equation.loss

            # (X): Simplify the equation with SymPy's simplify() applied to PySR's SymPy representation:
            simplified_equation = sp.simplify(equation.sympy_format)

            # (X): Convert the simplifed equation into LaTeX for plotting:
            equation_latex = sp.latex(simplified_equation)

            # (X): Use PySR results to predict the output:
            y_pysr_mean_predictions = py_regressor_models.predict(
                X = pseudodata_dataframe[independent_variable_dataframe_columns],
                index = index)

            # (X): Add a line plot for each prediction:
            plot_customization_pysr_predictions.add_line_plot(
                x_data = training_x_data,
                y_data = y_pysr_mean_predictions,
                label = fr"PySR (C = {equation_complexity}) (L = {equation_loss}): $y(x) = {equation_latex}$",
                color = color,
                linestyle = '-',
                alpha = 0.24,)
        
    if function_input_dimension == 2:
        
        axis_instance_pysr_predictions = figure_pysr_predictions.add_subplot(1, 1, 1, projection = '3d')

        plot_customization_pysr_predictions = PlotCustomizer(
            axis_instance_pysr_predictions,
            title = fr"Replica Method Predictions for $N = {NUMBER_OF_REPLICAS}$",
            xlabel = r"$x$",
            ylabel = r"$f(x)$",)
        
        plot_customization_pysr_predictions.add_surface_plot(
            x_data = training_x_grid_data,
            y_data = training_y_grid_data,
            z_data = y_mean.reshape(
                len(training_x_data['x_1'].unique()),
                len(training_x_data['x_2'].unique())
            ),
            label = r'Replica Average',
            colormap = "inferno",
            alpha = 0.7)
        
        # (4): Add data to the Axes Object:
        plot_customization_pysr_predictions.add_3d_error_scatter_plot(
            x_data = experimental_x_data[:, 0],
            y_data = experimental_x_data[:, 1],
            z_data = experimental_y_data,
            z_errors = experimental_y_error_data,
            label = "Experimental Data",
            color = "red",
            alpha = 0.8)
        
        for index, (equation, color) in enumerate(zip(sorted_equations, colors)):

            # (X): Extract the equation's complexity with `.complexity`:
            equation_complexity = equation.complexity

            # (X): Extract the loss during fitting with `.loss`:
            equation_loss = equation.loss

            # (X): Simplify the equation with SymPy's simplify() applied to PySR's SymPy representation:
            simplified_equation = sp.simplify(equation.sympy_format)

            # (X): Convert the simplifed equation into LaTeX for plotting:
            equation_latex = sp.latex(simplified_equation)

            x1_vals = np.sort(training_x_data.iloc[:, 0].unique())
            x2_vals = np.sort(training_x_data.iloc[:, 1].unique())
            x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals, indexing="ij")

            # Build prediction surface
            mesh_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

            # (X): Use PySR results to predict the output:
            y_pysr_mean_predictions = py_regressor_models.predict(
                X = pd.DataFrame(mesh_points, columns = training_x_data.columns),
                index = index)
            y_pysr_mean_predictions = y_pysr_mean_predictions.reshape(x1_grid.shape)

            plot_customization_pysr_predictions.add_surface_plot(
                x_data = x1_grid,
                y_data = x2_grid,
                z_data = y_pysr_mean_predictions,
                label = fr"PySR (C = {equation_complexity}) (L = {equation_loss}): $y(x_1, x_2) = {equation_latex}$",
                colormap = "viridis",
                alpha = 0.25)


    axis_instance_pysr_predictions.legend(
        loc = "center left",
        fontsize = 9,
        shadow = True,
        bbox_to_anchor = (1.05, 0.5),
        borderaxespad = 0.,
        frameon = True)

    plt.subplots_adjust(right=0.75)

    figure_pysr_predictions.tight_layout()

    figure_pysr_predictions.savefig(
        fname = f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/performance/replica_average_with_sr_v{_version_number}")
    plt.close()

    py_regressor_models.fit(
        pd.DataFrame(training_x_data),
        y_median)
    
    print(f"> Best fit model for this: {py_regressor_models.latex()}")

    figure_pysr_predictions = plt.figure(figsize = (18, 6))
    axis_instance_pysr_predictions = figure_pysr_predictions.add_subplot(1, 1, 1)
    plot_customization_pysr_predictions = PlotCustomizer(
        axis_instance_pysr_predictions,
        title = fr"Replica Method Medians Predictions for $N = {NUMBER_OF_REPLICAS}$",
        xlabel = r"$x$",
        ylabel = r"$f(x)$",)
    plot_customization_pysr_predictions.add_line_plot(
        x_data = training_x_data,
        y_data = y_median,
        label = "Replica Median",
        color = "blueviolet",
        linestyle = '-')
    plot_customization_pysr_predictions.add_fill_between_plot(
        x_data = training_x_data,
        lower_y_data = y_q1,
        upper_y_data = y_q3,
        label = "IQR",
        color = "lightgray",
        alpha = 0.34,)
    plot_customization_pysr_predictions.add_errorbar_plot(
        x_data = training_x_data,
        y_data = training_y_data,
        x_errorbars = np.zeros(y_error_data.shape),
        y_errorbars = y_error_data,
        label = "Experimental Data",
        color = "red",
        marker = 'o',)

    sorted_equations = sorted(py_regressor_models.equations_.itertuples(), key=lambda eq: eq.loss, reverse=True)

    colors = plt.cm.jet(np.linspace(0, 1, len(sorted_equations)))

    for index, (equation, color) in enumerate(zip(sorted_equations, colors)):

        equation_complexity = equation.complexity
        equation_loss = equation.loss
        # equation_latex = equation.sympy_format
        simplified_equation = sp.simplify(equation.sympy_format)  # Simplify the equation
        equation_latex = sp.latex(simplified_equation)  # Convert to LaTeX format

        y_pysr_median_predictions = py_regressor_models.predict(pd.DataFrame(training_x_data), index = index)

        plot_customization_pysr_predictions.add_line_plot(
            x_data = training_x_data,
            y_data = y_pysr_median_predictions,
            label = fr"PySR (C = {equation_complexity}) (L = {equation_loss}): $y(x) = {equation_latex}$",
            color = color,
            linestyle = '-',
            alpha = 0.24,)

    axis_instance_pysr_predictions.legend(
        loc="center left",
        fontsize=9,
        shadow=True,
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.,
        frameon=True
    )
    
    figure_pysr_predictions.tight_layout()

    figure_pysr_predictions.savefig(
        fname = f"{_PATH_SCIENCE_ANALYSIS}version_{_version_number}/plots/performance/replica_median_with_sr_v{_version_number}")
    plt.close()


if __name__ == "__main__":
    run()
