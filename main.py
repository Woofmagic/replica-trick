import numpy as np
import sympy as sp

from app.utilities.plotting.plot_data import construct_plot

from app.utilities.mathematics.sympy_function_generator import sympy_lambdify_expression
from app.utilities.mathematics.sympy_function_generator import sympy_generate_random_function

from app.utilities.mathematics.function_generator import generate_random_function

def plot_function():
    x_data = np.arange(1, 100, 0.05)
    sympy_symbol_x = sp.Symbol('x')
    DEPTH_PARAMETER = 2
    randomly_generated_sympy_function = sympy_generate_random_function(sympy_symbol_x, DEPTH_PARAMETER)
    randomly_generated_function = sympy_lambdify_expression(sympy_symbol_x, randomly_generated_sympy_function)
    y_data = randomly_generated_function(x_data)
    construct_plot(x_data, y_data, randomly_generated_sympy_function)

def run():
    print(f"> Now running...")
    plot_function()


if __name__ == "__main__":
    run()