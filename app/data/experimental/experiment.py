# External Library | NumPy
import numpy as np

# External Library | Matplotlib
import matplotlib.pyplot as plt

# External Library | SymPy
import sympy as sp

from app.utilities.plotting.plot_customizer import PlotCustomizer

from app.utilities.mathematics.statistics import sample_from_numpy_normal_distribution
from app.utilities.mathematics.sympy_function_generator import sympy_lambdify_expression
from app.utilities.mathematics.sympy_function_generator import sympy_generate_random_function

from app.utilities.writing.latex_generation import generate_document

class ExperimentalSetup:

    def __init__(
            self,
            experiment_name: str,
            number_of_data_points: int,
            underlying_function):

        _NUMBER_OF_DATA_POINTS_RICH = 5000
        _NUMBER_OF_DATA_POINTS_MEDIUM = 200
        _NUMBER_OF_DATA_POINTS_SPARSE = 40

        self.experiment_name = experiment_name

        self.number_of_data_points = number_of_data_points
        self.underlying_function = underlying_function

        self.independent_variable_values = np.array([])
        self.pure_experimental_values = np.array([])
        self.dependent_variable_values = np.array([])
        self.pandas_dataframe_of_experimental_data = None

        self._EXPERIMENTAL_START_VALUE = 1.
        self._EXPERIMENTAL_END_VALUE = 50.
        self._EXPERIMENTAL_SMEAR_STANDARD_DEVIATION = 0.192

    def do_experiment(self):
        """
        # Title: `do_experiment`

        ## Description: 
        We "do" the experiment, which means we plug-and-chug
        all the given values of the independent variable $x$ into
        the underlying function $f(x)$. 
        
        ## Parameters:
        Nothing!

        ## Returns:
        Nothing!
        """

        # (1): Calculate the range of the experiment:
        experimental_range = self._EXPERIMENTAL_END_VALUE - self._EXPERIMENTAL_START_VALUE

        # (2): Obtain the interval between experimentally sampled points via range/N:
        equidistant_points = experimental_range / self.number_of_data_points
        
        # (3): Obtain an iterable (list/array) of the explicit experimental values between the END and START values:
        self.independent_variable_values = np.arange(
            self._EXPERIMENTAL_START_VALUE,
            self._EXPERIMENTAL_END_VALUE,
            equidistant_points)

        # (4): Perform the plug-and-chugging of x into the generated f(x) as the first step:
        self.pure_experimental_values = self.underlying_function(self.independent_variable_values)

        # (5): Then, *add* the Gaussian noise on top of the "pure" f(x) value:
        self.dependent_variable_values = sample_from_numpy_normal_distribution(self.pure_experimental_values, self._EXPERIMENTAL_SMEAR_STANDARD_DEVIATION)

    def write_raw_data(self):
        """
        
        """

        import pandas as pd

        pandas_series_of_independent_variables = pd.Series(self.independent_variable_values)
        pandas_series_of_dependent_variables = pd.Series(self.dependent_variable_values)

        pandas_dataframe_of_experimental_data = pd.DataFrame(
            {
                "x": pandas_series_of_independent_variables,
                "y": pandas_series_of_dependent_variables
            })

        pandas_dataframe_of_experimental_data.to_csv(f'{self.experiment_name}_raw_data.csv')

        self.pandas_dataframe_of_experimental_data = pandas_dataframe_of_experimental_data

    def plot_experimental_data(self):
        """
        ## Description:
        When a big experiment finishes, they always construct plots
        to see what is going on. Here, we construct plots using the data
        we just took.
        
        ## Parameters:
        Nothing!

        ## Returns:
        Nothing!
        """
        # (1): Set up the Figure instance
        figure_instance = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance = figure_instance.add_subplot(1, 1, 1)
        
        # (3): Customize the Axes Object:
        plot_customization = PlotCustomizer(
            axis_instance,
            title = r"Experiment {{self.experiment_name}}",
            xlabel = r"$x$",
            ylabel = r"$f(x)$")
        
        # (4): Add data to the Axes Object:
        plot_customization.add_errorbar_plot(
            x_data = self.independent_variable_values,
            y_data = self.dependent_variable_values,
            x_errorbars = np.array([0.]),
            y_errorbars = [self._EXPERIMENTAL_SMEAR_STANDARD_DEVIATION for item in range(len(self.dependent_variable_values))],
            label = r'Experimental Data',
            color = 'red')
        
        # (7): Show the plot for the time being:
        figure_instance.savefig(f'{self.experiment_name}_raw_data.png')

    def plot_underlying_function(
            self,
            underlying_symbolic_function: sp.FunctionClass,
            underlying_function):
        """
        ## Description:
        We plot the continuous, underlying function that
        the experiment was seeking to understand.
        
        ## Parameters:
        Nothing!

        ## Returns:
        Nothing!
        """

        # (1): Set up the Figure instance
        figure_instance = plt.figure(figsize = (18, 6))

        # (2): Add an Axes Object:
        axis_instance = figure_instance.add_subplot(1, 1, 1)
        
        # (3): Customize the Axes Object:
        plot_customization = PlotCustomizer(
            axis_instance,
            title = r"Underlying Function: $f(x) = {}$".format(sp.latex(underlying_symbolic_function)),
            xlabel = r"$x$",
            ylabel = r"$f(x)$")
        
        # (4): Construct an iterable with values to plug into the underlying function for a smooth plot:
        x_data_range = np.arange(min(self.independent_variable_values), max(self.independent_variable_values), 0.01)

        # (5): Add a line plot for the underlying function:
        plot_customization.add_line_plot(
            x_data_range, 
            underlying_function(x_data_range),
            color = 'black')
        
        # (7): Show the plot for the time being:
        figure_instance.savefig('underlying_function_v4')

def conduct_experiment(
        experiment_name: str):
    """
    ## Description:
    This function actually initiates the entire "experiment." 
    
    ## Parameters:
    Nothing!

    ## Returns:
    Nothing!

    ## Examples:

    ## Notes:
    """

    # (1): First, we determine how robust and serious our experiment is:
    number_of_data_points = 45

    # (2): We need to define a Sympy variable "x" that's our independent variable:
    sympy_symbol_x = sp.Symbol('x')

    # (3): We now specify how "difficult" our underlying function will be:
    DEPTH_PARAMETER = 3

    # (4): Next, we generate the underlying function (symbolically, in Sympy):
    underlying_symbolic_function = sympy_generate_random_function(sympy_symbol_x, DEPTH_PARAMETER)

    # (5): We obtain a "Python understandable" function of the symbolic function above:
    underlying_function = sympy_lambdify_expression(sympy_symbol_x, underlying_symbolic_function)

    # (6): Finally, we set up the Experiment:
    experiment_instance = ExperimentalSetup(
        experiment_name,
        number_of_data_points, 
        underlying_function)

    # (7): We then conduct the experiment:
    experiment_instance.do_experiment()

    # (8): When the experiment has finished, we write a file containing the raw data:
    experiment_instance.write_raw_data()

    # (9): Once the experiment has finished, we construct the "raw data" plot (contains uncertainty)
    experiment_instance.plot_experimental_data()

    # (10): We also provide the plot that contains the "underlying function" that we're trying to probe:
    experiment_instance.plot_underlying_function(
        underlying_symbolic_function,
        underlying_function)

    print(experiment_instance.pandas_dataframe_of_experimental_data.to_latex(
        buf = None,
        header = [r"$x$", r"$y$"],
        index = True,
        na_rep = "NaN",
        escape = False,
        column_format = "|" + "c|" * len(experiment_instance.pandas_dataframe_of_experimental_data.columns)

    ))

    generate_document(
        underlying_equation = sp.latex(underlying_symbolic_function),
        experimental_data_table = experiment_instance.pandas_dataframe_of_experimental_data.to_latex(),
        experiment_name = f"E{experiment_name}")

    experimental_x_data = experiment_instance.independent_variable_values
    experimental_y_data = experiment_instance.dependent_variable_values

    return experimental_x_data, experimental_y_data
