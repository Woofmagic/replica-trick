# Native Library | datetime
import datetime

# Native Library | os
import os

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

# External Library | PySR
from pysr import PySRRegressor

_version_number = 1

_SEARCH_SPACE_BINARY_OPERATORS = [
    "+", "-", "*", "/", "^"
]

_SEARCH_SPACE_UNARY_OPERATORS = [
   "exp", "log", "sqrt", "sin", "cos", "tan"
]

_SEARCH_SPACE_MAXIUMUM_COMPLEXITY = 20

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
    niterations = 40,

    # (2): The number of "populations" running:
    populations = 15,

    # (3): The size of each population:
    population_size = 33,

    # (4): Whatever the fuck this means:
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
    parsimony = 0.0032,

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

    if not os.path.exists(base_path):
        os.makedirs(base_path)  # Ensure the base path exists
    
    # Regex to match 'version_x' pattern
    version_pattern = re.compile(r'version_(\d+)')
    
    existing_versions = []
    for entry in os.listdir(base_path):
        match = version_pattern.match(entry)
        if match:
            existing_versions.append(int(match.group(1)))
    
    next_version = max(existing_versions, default=-1) + 1
    return next_version

def run():
    
    _PATH_SCIENCE_ANALYSIS = 'science/analysis/'
    _PATH_SCIENCE_DATA = 'science/data'

    # Get next version directories
    _version_number = get_next_version(_PATH_SCIENCE_ANALYSIS)

    print(f"> Determined next analysis directory: {_version_number}")

    os.mkdir(f'{_PATH_SCIENCE_ANALYSIS}/version_{_version_number}/fits')
    os.mkdir(f'{_PATH_SCIENCE_ANALYSIS}/version_{_version_number}/plots')
    os.mkdir(f'{_PATH_SCIENCE_ANALYSIS}/version_{_version_number}/sr_analysis')

    try:
        # tf.config.set_visible_devices([],'GPU')
        tensorflow_found_devices = tf.config.list_physical_devices()

        if len(tf.config.list_physical_devices()) != 0:
            for device in tensorflow_found_devices:
                print(f"> TensorFlow detected device: {device}")

        else:
            print("> TensorFlow didn't find CPUs or GPUs...")

    except Exception as error:
        print(f"> TensorFlow could not find devices due to error:\n> {error}")

    print(f"> Now running TensorFlow Version {tf.version.VERSION}")

    # (1): Nonsense foor now
    kinematic_set_integer = 1

    # (2): The number of replicas to use.
    number_of_replicas = 50
    EPOCHS = 2000

    # run_replica_method(kinematic_set_integer, number_of_replicas)

    training_x_data, training_y_data = conduct_experiment()

    # (1): Begin iterating over the replicas:
    for replica_index in range(number_of_replicas):

        initializer = tf.keras.initializers.RandomUniform(
            minval = -10.0,
            maxval = 10.0,
            seed = None)

        input_x_value = Input(shape = (1, ), name = 'input_layer')
        
        # (3): Define the Model Architecture:
        x1 = Dense(256, activation = "relu6", kernel_initializer = initializer)(input_x_value)
        x2 = Dense(256, activation = "relu6", kernel_initializer = initializer)(x1)
        x3 = Dense(256, activation = "relu6", kernel_initializer = initializer)(x2)
        x4 = Dense(256, activation = "relu6", kernel_initializer = initializer)(x3)
        x5 = Dense(256, activation = "relu6", kernel_initializer = initializer)(x4)
        output_y_value = Dense(1, activation = "linear", kernel_initializer = initializer, name = 'output_y_value')(x5)

        # (4): Define the model as as Keras Model:
        tensorflow_network = Model(inputs = input_x_value, outputs = output_y_value, name = "basic_function_predictor")
        
        tensorflow_network.compile(
            optimizer='adam',
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [
                tf.keras.metrics.MeanSquaredError()
                ])
        
        tensorflow_network.summary()

        start_time_in_milliseconds = datetime.datetime.now().replace(microsecond = 0)
        
        print(f"> Replica #{replica_index + 1} now running...")

        history_of_training_7 = tensorflow_network.fit(
            training_x_data, 
            training_y_data, 
            epochs = EPOCHS)

        # (3): Construct the loss plot:
        training_loss_data_7 = history_of_training_7.history['loss']
        model_predictions_7 = tensorflow_network.predict(training_x_data)

        tensorflow_network.save(f"replica_number_{replica_index + 1}_v{_version_number}.keras")

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
            title = r"Neural Network Loss per Epoch",
            xlabel = r"Epoch",
            ylabel = r"Loss (MSE)")
        
        plot_customization_nn_loss.add_line_plot(
            x_data = np.arange(0, EPOCHS, 1),
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
            x_data = training_x_data,
            y_data = training_y_data,
            label = r'Experimental Data',
            color = "red")
        
        plot_customization_data_comparison.add_scatter_plot(
            x_data = training_x_data,
            y_data = model_predictions_7,
            label = r'Model Predictions',
            color = "orange")
        
        figure_instance_nn_loss.savefig(f"loss_v{replica_index+1}_v{_version_number}")
        figure_instance_fitting.savefig(f"fitting_replica_{replica_index+1}_v{_version_number}")

    model_paths = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd()) if file.endswith(f"v{_version_number}.keras")]
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

    y_mean, y_min, y_max, y_percentile_10, y_percentile_20, y_percentile_30, y_percentile_40, y_percentile_60, y_percentile_70, y_percentile_80, y_percentile_90 = predict_with_models(models, training_x_data)

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
    
    figure_instance_predictions.savefig(f"replica_average_data_v{_version_number}")
    plt.close()

    py_regressor_models.fit(training_x_data.reshape(-1, 1), y_mean)
    print(py_regressor_models.sympy())
    py_regressor_models.sympy()
    py_regressor_models.latex()
    py_regressor_models.latex_table()



if __name__ == "__main__":
    run()