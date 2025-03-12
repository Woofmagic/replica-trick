from app.utilities.plotting.plot_customizer import PlotCustomizer

# External Library | NumPy
import numpy as np

import os

# External Library | Matplotlib
import matplotlib.pyplot as plt
# rcParams stuff....
plt.rcParams.update(plt.rcParamsDefault)

# External Library | TensorFlow
import tensorflow as tf

# External Library | PySR
from pysr import PySRRegressor

# External Library | Pandas
import pandas as pd

NUMBER_OF_REPLICAS = 43

_SEARCH_SPACE_BINARY_OPERATORS = [
    "+", "-", "*", "/", "^"
]

_SEARCH_SPACE_UNARY_OPERATORS = [
   "exp", "log", "sqrt", "sin", "cos", "tan"
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
    niterations = 500,

    # (2): The number of "populations" running:
    populations = 500,

    # (3): The size of each population:
    population_size = 100,

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

    # (1): Multiplicative factor that penalizes a complex function: l(E) = l_{loss}(E) exp(parsimony * etc.)
    parsimony = 0.0032,

    # (2): A complicated dictionary governing how complex a given operation can be:
    constraints = None,

    # (3): Another dictionary that enforces the number of times an operator may be nested:
    nested_constraints = None,

    # (4): Another dictionary that limits the complexity per operator:
    complexity_of_operators = None)


for _version_number in range(9, 13):
    model_paths = [os.path.join(os.getcwd(), f"app/science/data/version_{_version_number}/replicas/{file}") for file in os.listdir(f"app/science/data/version_{_version_number}/replicas") if file.endswith(".keras")]
    models = [tf.keras.models.load_model(path) for path in model_paths]

    print(f"> Obtained {len(models)} models!")

    raw_data = pd.read_csv(f"app/science/data/version_{_version_number}/raw/E{_version_number}_raw_data.csv")
    training_x_data = raw_data['x']
    training_y_data = raw_data['y']
    y_error_data = raw_data['y_error']

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
        title = r"$N = {}$".format(NUMBER_OF_REPLICAS),
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
    
    plot_customization_predictions.add_errorbar_plot(
            x_data = training_x_data,
            y_data = training_y_data,
            x_errorbars = np.zeros(y_error_data.shape),
            y_errorbars = y_error_data,
            label = r'Experimental Data',
            color = "red",
            marker = 'o',)
    
    figure_instance_predictions.savefig(f"replica_average_data_v{_version_number}")
    plt.close()

    py_regressor_models.fit(pd.DataFrame(training_x_data), y_mean)
    print(py_regressor_models.sympy())
    py_regressor_models.sympy()
    py_regressor_models.latex()
    py_regressor_models.latex_table()