import numpy as np
import sympy as sp

# Functions | Constant:
def sympy_constant_function(
    sympy_variable_x: sp.Symbol, 
    parameter_A: float) -> float:

    try:
        return parameter_A * sympy_variable_x
    except:
        FLOAT_ZERO = 0.
        return FLOAT_ZERO

# Functions | Polynomial:
def sympy_nth_degree_polynomial(
    sympy_variable_x: sp.Symbol, 
    list_of_polynomial_coefficients: list) -> float:

    try:
        result = 0
        for i, coefficient in enumerate(list_of_polynomial_coefficients):
            result += coefficient * (sympy_variable_x ** i)
        return result
    except:
        FLOAT_ZERO = 0.
        return FLOAT_ZERO

# Functions | Logarithm:
def sympy_logarithmic_function(
    sympy_variable_x: sp.Symbol, 
    parameter_A: float,
    parameter_B: float) -> float:

    try:
        return parameter_A * sp.log(parameter_B * sympy_variable_x)
    except Exception as ERROR:
        print(f"Fuck")
        return 0.

# Functions | Exponential:
def sympy_exponential_function(
        sympy_variable_x: sp.Symbol,
        parameter_A: float,
        parameter_B: float) -> float:
    """
    # Description:
    --------------
    Calculate the exponential function A * e^(B * x).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The scaling factor for the output.

    :param B: The scaling factor for the input.

    # Returns
    --------------
    :return: The result of the exponential function.
    """
    try:
        return parameter_A * sp.exp(parameter_B * sympy_variable_x)
    except Exception as ERROR:
        print(f"Fuck")
        return 0.

# Functions | Sine:
def sympy_sine_function(
        sympy_variable_x: sp.Symbol,
        parameter_A: float,
        parameter_B: float,
        parameter_C: float) -> float:
    """
    # Description:
    --------------
    Calculate the sinusoidal function A * sin(B * x + C).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The amplitude of the sine wave.

    :param B: The frequency scaling factor.

    :param C: The phase shift.

    # Returns
    --------------
    :return: The result of the sinusoidal function.
    """
    try:
        return parameter_A * sp.sin(parameter_B * sympy_variable_x + parameter_C)
    except Exception as ERROR:
        return 0.
    
# Functions | Cosine:
def sympy_cosine_function(
        sympy_variable_x: sp.Symbol,
        parameter_A: float,
        parameter_B: float,
        parameter_C: float) -> float:
    """
    # Description:
    --------------
    Calculate the cos function A * cos(B * x + C).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The amplitude of the sine wave.

    :param B: The frequency scaling factor.

    :param C: The phase shift.

    # Returns
    --------------
    :return: The result of the cosine function.
    """
    try:
        return parameter_A * sp.cos(parameter_B * sympy_variable_x + parameter_C)
    except Exception as ERROR:
        return 0.
    
# Functions | Tangent:
def sympy_tangent_function(
        sympy_variable_x: sp.Symbol,
        parameter_A: float,
        parameter_B: float,
        parameter_C: float) -> float:
    """
    # Description:
    --------------
    Calculate the tangent function A * tan(B * x + C).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The amplitude of the sine wave.

    :param B: The frequency scaling factor.

    :param C: The phase shift.

    # Returns:
    --------------
    :return: The result of the tangent function.
    """
    try:
        return parameter_A * sp.tan(parameter_B * sympy_variable_x + parameter_C)
    except Exception as ERROR:
        return 0.
    
def sympy_generate_random_function(
        sympy_variable_x: sp.Symbol,
        depth: int) -> float:
    
    functions = [
        sympy_constant_function,
        sympy_exponential_function,
        sympy_logarithmic_function,
        sympy_sine_function,
        sympy_cosine_function,
        sympy_tangent_function]

    result = sympy_variable_x
    
    for iteration in range(depth):
        function_index = np.random.randint(0, len(functions))
        function = functions[function_index]
        number_of_arguments_per_function = function.__code__.co_argcount
        function_parameters = np.round(np.random.uniform(-5, 5, size = number_of_arguments_per_function - 2))
        result = function(result, 1, *function_parameters)

    return result

def sympy_lambdify_expression(
        sympy_variable_x: sp.Symbol,
        sympy_expression: sp.FunctionClass):
    return sp.lambdify(sympy_variable_x, sympy_expression, 'numpy')