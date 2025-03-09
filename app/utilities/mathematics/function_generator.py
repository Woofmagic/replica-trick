import numpy as np

# Functions | Polynomial:
def nth_degree_polynomial(
        x: float, 
        list_of_polynomial_coefficients: list) -> float:
    """
    # Description
    --------------
    Calculate the value of an N-th degree polynomial at x.
    
    # Parameters
    --------------
    :param x: The input value where the polynomial is evaluated.

    :param coefficients: A list of coefficients [a_0, a_1, ..., a_N] where 
        a_i is the coefficient for the x^i term.

    # Returns
    --------------

    :return: The value of the polynomial at x.
    """
    try:
        result = 0
        for i, coefficient in enumerate(list_of_polynomial_coefficients):
            result += coefficient * (x ** i)
        return result
    except:
        FLOAT_ZERO = 0.
        return FLOAT_ZERO

# Functions | Logarithm:
def logarithmic_function(x: float, parameter_A: float, parameter_B: float):
    """
    # Description
    --------------
    Calculate the logarithmic function A * ln(B * x).
    
    # Parameters
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The scaling factor for the output.

    :param B: The scaling factor for the input.

    # Returns
    --------------
    :return: The result of the logarithmic function.
    """
    
    try:
        return parameter_A * np.log(parameter_B * x)
    except Exception as ERROR:
        return 0.

# Functions | Exponential:
def exponential_function(x: float, parameter_A: float, parameter_B: float) -> float:
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
        return parameter_A * np.exp(parameter_B * x)
    except Exception as ERROR:
        return 0.

# Functions | Sine:
def sine_function(x: float, parameter_A: float, parameter_B: float, parameter_C: float) -> float:
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
        return parameter_A * np.sin(parameter_B * x + parameter_C)
    except Exception as ERROR:
        return 0.
    
# Functions | Cosine:
def cosine_function(x: float, parameter_A: float, parameter_B: float, parameter_C: float) -> float:
    """
    # Description:
    --------------
    Calculate the cosine function A * cos(B * x + C).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The amplitude of the sine wave.

    :param B: The frequency scaling factor.

    :param C: The phase shift.

    # Returns:
    --------------
    :return: The result of the cosine function.
    """
    try:
        return parameter_A * np.cos(parameter_B * x + parameter_C)
    except Exception as ERROR:
        return 0.

# Functions | Tangent:
def tangent_function(x: float, parameter_A: float, parameter_B: float, parameter_C: float) -> float:
    """
    # Description:
    --------------
    Calculate the sinusoidal function A * tan(B * x + C).
    
    # Parameters:
    --------------
    :param x: The input value(s) where the function is evaluated (can be a numpy array).

    :param A: The amplitude of the sine wave.

    :param B: The frequency scaling factor.

    :param C: The phase shift.

    # Returns:
    --------------
    :return: The result of the sinusoidal function.
    """
    try:
        return parameter_A * np.tan(parameter_B * x + parameter_C)
    except Exception as ERROR:
        return 0.
    
def generate_random_function(x_data, depth: int) -> float:
    
    functions = [exponential_function, logarithmic_function, sine_function, cosine_function]

    result = x_data.copy()
    
    for iteration in range(depth):
        func_idx = np.random.randint(0, len(functions)) 
        function = functions[func_idx]
        num_args = function.__code__.co_argcount 
        print(f"> Number of arguments of {function.__name__} is {num_args}")
        params = np.random.uniform(1, 1, size = num_args - 2)  # Generate random parameters
        print(params)
        result = function(result, 1, *params)  # Apply the chosen function to the result using *params

    print(result)
    return result