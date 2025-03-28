# External Library | NumPy
import numpy as np

# External Library | Matplotlib
import matplotlib.pyplot as plt

# External Library | SymPy
import sympy as sp

# External Library | Pandas:
import pandas as pd

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
        """
        Initialize the experimental setup.
        """
        self.experiment_name = experiment_name
        self.number_of_data_points = number_of_data_points
        self.underlying_function = underlying_function

        self._EXPERIMENTAL_START_VALUE = -1.0
        self._EXPERIMENTAL_END_VALUE = 1.0
        self._BASE_SMEAR_STANDARD_DEVIATION = 0.052

        self._SYSTEMATIC_SHIFT_STD = 0.05  # Bias-induced shift in measurement
        self._STOCHASTIC_NOISE_LOW = 0.09
        self._STOCHASTIC_NOISE_HIGH = 0.14  # Random uniform noise
        self._WHITE_NOISE_TARGET_LEVEL = 0.09

        self._INCREASE_ERRORS_AT_EDGES = False  # More uncertainty at edges
        self._INCREASE_ERRORS_AT_PEAKS = False  # More uncertainty at high function values

        self._NUMBER_OF_DATA_POINTS_RICH = 1000
        self._NUMBER_OF_DATA_POINTS_MEDIUM = 200
        self._NUMBER_OF_DATA_POINTS_SPARSE = 40
        self._USING_EQUIDISTANT_POINTS = True

        self.independent_variable_values = np.array([])
        self.pure_experimental_values = np.array([])
        self.dependent_variable_values = np.array([])
        self.experimental_errors = np.array([])
        self.pandas_dataframe_of_experimental_data = None

    def _generate_equidistant_x_values(self):
        """
        Generate equidistant x-values.
        """
        # (1): Calculate the range of the experiment:
        experimental_range = self._EXPERIMENTAL_END_VALUE - self._EXPERIMENTAL_START_VALUE

        equidistant_points = experimental_range / self.number_of_data_points

        return np.sort(np.arange(
            self._EXPERIMENTAL_START_VALUE,
            self._EXPERIMENTAL_END_VALUE,
            equidistant_points))

    def _generate_nonuniform_x_values(self):
        """
        Generate x values with non-uniform spacing.
        """
        return np.sort(
            np.random.uniform(
                low = self._EXPERIMENTAL_START_VALUE,
                high = self._EXPERIMENTAL_END_VALUE,
                size = self.number_of_data_points))
    
    def _generate_variable_errors(self):
        """
        Generate realistic varying errors for each data point.
        """
        # _ALPHA = 0.02
        # _BETA = 0.05
        # _GAMMA = 0.5

        # sigma_y = _BETA * (self.dependent_variable_values.max() - self.dependent_variable_values.min())
        # rand_y = np.random.normal(0, 1, len(self.dependent_variable_values))
        # dy = sigma_y * rand_y
        # dy = np.maximum(dy, sigma_y)

        dy = np.abs(np.random.normal(self._BASE_SMEAR_STANDARD_DEVIATION, 0.))
        return dy

    def _generate_experimental_data(self):
        """
        Generate realistic varying errors for each data point.
        """
        # (1): Add a Gaussian smear to the pure (true) values --- this turns true data into "experimental data":
        gaussian_component = np.random.normal(self.pure_experimental_values, self._BASE_SMEAR_STANDARD_DEVIATION)

        # (2): Compute a Gaussian noise to be added on top of the experimental data:
        # stochastic_component = np.random.uniform(self._STOCHASTIC_NOISE_LOW, self._STOCHASTIC_NOISE_HIGH, size = len(self.pure_experimental_values))
        stochastic_component = np.random.uniform(-self._STOCHASTIC_NOISE_LOW, self._STOCHASTIC_NOISE_LOW, size = len(self.pure_experimental_values))
        # stochastic_component = np.random.normal(
        #     loc = 0., 
        #     # Below comes from EquATION ?? on pg. 16 from: https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/pdf/nihms-1824079.pdf
        #     scale = self._WHITE_NOISE_TARGET_LEVEL * np.sqrt(len(self.pure_experimental_values) * np.sum(self.pure_experimental_values**2)),
        #     size = len(self.pure_experimental_values))

        # # (3): If our "detector" has problems with systematics at the edges of the experimental phase space...:
        # if self._INCREASE_ERRORS_AT_EDGES:

        #     # (3.1): 
        #     edge_factor = np.exp(-((self.independent_variable_values - np.mean(self.independent_variable_values))**2) / 50)

        #     # (3.2): Inflate the 
        #     gaussian_component += edge_factor * 0.1 

        # # (4): If our "detector" has issues with peaks of a function (resonance, etc.)...:
        # if self._INCREASE_ERRORS_AT_PEAKS:

        #     # (4.1): 
        #     peak_factor = np.exp(-self.pure_experimental_values / np.max(self.pure_experimental_values + 1e-6))

        #     # (4.2): Inflate the peak by a factor:
        #     gaussian_component += peak_factor * 0.09

        # Ensure non-negative errors
        return gaussian_component + stochastic_component

    def do_experiment(self):
        """
        ## Description: 
        We "do" the experiment, which means we plug-and-chug
        all the given values of the independent variable $x$ into
        the underlying function $f(x)$. 
        
        ## Parameters:
        Nothing!

        ## Returns:
        Nothing!
        """

        # (1.1): If we are using equally-spaced x-values:
        if self._USING_EQUIDISTANT_POINTS:

            # (1.1.1): Obtain an iterable (list/array) of the explicit experimental values between the END and START values:
            self.independent_variable_values = self._generate_equidistant_x_values()

        # (1.2): If we are doing things realistically:
        else:

            # (1.2.1): ...
            self.independent_variable_values = self._generate_nonuniform_x_values()

        # (2): Perform the plug-and-chugging of x into the generated f(x) as the first step:
        self.pure_experimental_values = self.underlying_function(self.independent_variable_values)

        # (3): Generate systematic shift
        systematic_shift = np.random.normal(0, self._SYSTEMATIC_SHIFT_STD, size = len(self.pure_experimental_values))

        # (4): Generate realistic experimental data:
        self.dependent_variable_values = self._generate_experimental_data()

        # (5): Generate realistic error bars:
        self.experimental_errors = self._generate_variable_errors()
        
    def write_raw_data(self):
        """
        ## Description:
        Save the generated experimental data to a CSV file.
        """
        pandas_dataframe_of_experimental_data = pd.DataFrame(
            data = {
                "x": self.independent_variable_values,
                "y": self.dependent_variable_values,
                "y_error": self.experimental_errors
            })

        pandas_dataframe_of_experimental_data.to_csv(f'E{self.experiment_name}_raw_data.csv', index_label = 'index')
        self.pandas_dataframe_of_experimental_data = pandas_dataframe_of_experimental_data

    def plot_experimental_data(self, underlying_function):
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
            title = "E{} Raw Data".format(self.experiment_name),
            xlabel = r"$x$",
            ylabel = r"$f(x)$")
        
        # (4): Add data to the Axes Object:
        plot_customization.add_errorbar_plot(
            x_data = self.independent_variable_values,
            y_data = self.dependent_variable_values,
            x_errorbars = np.array([0.]),
            y_errorbars =  self.experimental_errors,
            label = r'Experimental Data',
            color = 'red')
        
        # (5): Add data to the Axes Object:
        x_data_range = np.arange(min(self.independent_variable_values), max(self.independent_variable_values), 0.01)
        plot_customization.add_line_plot(
            x_data_range, 
            underlying_function(x_data_range),
            label = 'Underlying Function',
            color = 'gray')
        
        # (7): Show the plot for the time being:
        figure_instance.savefig(f'E{self.experiment_name}_raw_data.png')

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
        figure_instance.savefig(f'underlying_function_E{self.experiment_name}.png')

def conduct_experiment(experiment_name: str):
    """
    ## Description:
    This function actually initiates the entire "experiment." 
    
    ## Parameters:
    - experiment_name (str)

    ## Returns:
    Nothing!

    ## Examples:

    ## Notes:
    """

    # (1): First, we determine how robust and serious our experiment is:
    number_of_data_points = 100

    # (2): We need to define a Sympy variable "x" that's our independent variable:
    sympy_symbol_x = sp.Symbol('x')

    # (3): We now specify how "difficult" our underlying function will be:
    DEPTH_PARAMETER = 2

    # (4): Next, we generate the underlying function (symbolically, in Sympy):
    # underlying_symbolic_function = sympy_generate_random_function(sympy_symbol_x, DEPTH_PARAMETER)
    # # Linear:
    # underlying_symbolic_function = 0.65 * sympy_symbol_x - 0.18
    # # Quadratic
    # underlying_symbolic_function = 1.02 * sympy_symbol_x**2 - 2.78 * sympy_symbol_x + 3.4
    # # Lorentzian:
    underlying_symbolic_function = 1. / (sp.pi * 0.121 * (1. + ((sympy_symbol_x - (-0.117)) /  0.121)**2))
    # # Gaussian:
    # underlying_symbolic_function = sp.exp(- (sympy_symbol_x - 0.145)**2 / (0.214)**2) / (0.214 * sp.sqrt(2. * sp.pi))
    # Sigmoid:
    # a1, b1 = 2.5, 0.1  # Adjust steepness and center shift
    # underlying_symbolic_function = 3 / (1 + sp.exp(-a1 * (sympy_symbol_x - b1)))
    # # Bimodal Gaussian-like function:
    # a2, b2, c2 = 2.0, -0.5, 0.5
    # underlying_symbolic_function = a2 * (sp.exp(-((sympy_symbol_x - b2) / 0.3) ** 2) + sp.exp(-((sympy_symbol_x - c2) / 0.3) ** 2))
    # # Cosine:
    # a3, b3 = 2.0, 0.75
    # underlying_symbolic_function = a3 * sp.cos(b3 * sp.pi * sympy_symbol_x)

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
    experiment_instance.plot_experimental_data(underlying_function)

    # (10): We also provide the plot that contains the "underlying function" that we're trying to probe:
    experiment_instance.plot_underlying_function(
        underlying_symbolic_function,
        underlying_function)

    # print(experiment_instance.pandas_dataframe_of_experimental_data.to_latex(
    #     buf = None,
    #     header = [r"$x$", r"$y$", r"$\sigma_{y}$"],
    #     index = True,
    #     na_rep = "NaN",
    #     escape = False,
    #     column_format = "|" + "c|" * len(experiment_instance.pandas_dataframe_of_experimental_data.columns)

    # ))

    # generate_document(
    #     underlying_equation = sp.latex(underlying_symbolic_function),
    #     experimental_data_table = experiment_instance.pandas_dataframe_of_experimental_data.to_latex(
    #         buf = None,
    #     header = [r"$x$", r"$y$", r"$\sigma_{y}$"],
    #     index = True,
    #     na_rep = "NaN",
    #     escape = False,
    #     column_format = "|" + "c|" * len(experiment_instance.pandas_dataframe_of_experimental_data.columns)
    #     ),
    #     experiment_name = f"E{experiment_name}")

    experimental_x_data = experiment_instance.independent_variable_values
    experimental_y_data = experiment_instance.dependent_variable_values
    experimental_y_error_data = experiment_instance.experimental_errors

    return experimental_x_data, experimental_y_data, experimental_y_error_data
