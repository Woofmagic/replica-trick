import numpy as np
import sympy as sp

# from app.utilities.plotting.plot_data import construct_plot

# from app.utilities.mathematics.sympy_function_generator import sympy_lambdify_expression
# from app.utilities.mathematics.sympy_function_generator import sympy_generate_random_function

# from app.utilities.mathematics.function_generator import generate_random_function

from app.data.experimental.experiment import conduct_experiment

def run():
    print(f"> Now running...")

    x_data, y_data = conduct_experiment()


if __name__ == "__main__":
    run()