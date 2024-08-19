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

class ExperimentalSetup:

    def __init__(
            self,
            number_of_data_points: int,
            underlying_function):

        self.number_of_data_points = number_of_data_points
        self.underlying_function = underlying_function

        self.independent_variable_values = np.array([])
        self.pure_experimental_values = np.array([])
        self.dependent_variable_values = np.array([])

        self._EXPERIMENTAL_START_VALUE = 1.
        self._EXPERIMENTAL_END_VALUE = 10.
        self._EXPERIMENTAL_SMEAR_STANDARD_DEVIATION = 0.5

    def do_experiment(self):
        """
        # Description
        --------------
        We "do" the experiment, which means we plug-and-chug
        all the given values of the independent variable $x$ into
        the underlying function $f(x)$. 
        
        # Parameters
        --------------
        Nothing!

        # Returns
        --------------
        Nothing!
        """

        equidistant_points = (self._EXPERIMENTAL_END_VALUE - self._EXPERIMENTAL_START_VALUE) / self.number_of_data_points
        
        self.independent_variable_values = np.arange(
            self._EXPERIMENTAL_START_VALUE,
            self._EXPERIMENTAL_END_VALUE,
            equidistant_points
        )

        self.pure_experimental_values = self.underlying_function(self.independent_variable_values)
        self.dependent_variable_values = sample_from_numpy_normal_distribution(self.pure_experimental_values, self._EXPERIMENTAL_SMEAR_STANDARD_DEVIATION)

    def plot_experimental_data(
            self,
            underlying_symbolic_function: sp.FunctionClass,
            underlying_function):
        """
        # Title: `plot_experimental_data`

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
            title = r"Function: $f(x) = {}$".format(sp.latex(underlying_symbolic_function)),
            xlabel = r"$x$",
            ylabel = r"$f(x)$")
        
        # (4): Add data to the Axes Object:
        plot_customization.add_errorbar_plot(
            x_data = self.independent_variable_values, 
            y_data = self.dependent_variable_values,
            y_errorbars = [0 for item in range(len(self.dependent_variable_values))],
            x_errorbars = [0 for item in range(len(self.dependent_variable_values))],
            color = 'red')
        
        # (5): Construct an iterable with values to plug into the underlying function for a smooth plot:
        x_data_range = np.arange(min(self.independent_variable_values), max(self.independent_variable_values), 0.01)

        # (6): Add a line plot for the underlying function:
        plot_customization.add_line_plot(
            x_data_range, 
            underlying_function(x_data_range),
            color = 'crimson')
        
        # (7): Show the plot for the time being:
        plt.show()



def conduct_experiment():
    """
    # Description
    --------------
    This function actually initiates the entire "experiment." 
    
    # Parameters
    --------------
    Nothing!

    # Returns
    --------------
    Nothing!
    """

    # (1): First, we determine how robust and serious our experiment is:
    number_of_data_points = 50

    # (2): We need to define a Sympy variable "x" that's our independent variable:
    sympy_symbol_x = sp.Symbol('x')

    # (3): We now specify how "difficult" our underlying function will be:
    DEPTH_PARAMETER = 2

    # (4): Next, we generate the underlying function (symbolically, in Sympy):
    underlying_symbolic_function = sympy_generate_random_function(sympy_symbol_x, DEPTH_PARAMETER)

    # (5): We obtain a "Python understandable" function of the symbolic function above:
    underlying_function = sympy_lambdify_expression(sympy_symbol_x, underlying_symbolic_function)

    # (6): Finally, we set up the Experiment:
    experiment_instance = ExperimentalSetup(number_of_data_points, underlying_function)

    # (7): We then conduct the experiment:
    experiment_instance.do_experiment()
    experiment_instance.plot_experimental_data(underlying_symbolic_function, underlying_function)
