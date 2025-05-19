
import random
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
    degree_of_polynomial: int,
    *coefficients) -> float:
    """
    ## Description:
    Create an nth-degree polynomial function of x.

    ## Parameters:
    - sympy_variable_x (sp.Symbol): The symbolic variable.
    - n (int): The degree of the polynomial.
    - *coefficients (float): The polynomial coefficients in order [a_0, a_1, ..., a_n].

    ## Returns:
    - sp.Expr: The symbolic polynomial expression.
    """
    if len(coefficients) != degree_of_polynomial + 1:
        raise ValueError(f"Expected {degree_of_polynomial + 1} coefficients, but got {len(coefficients)}.")
    
    try:
        return sum(coefficients[i] * sympy_variable_x**i for i in range(degree_of_polynomial + 1))
    except:
        FLOAT_ZERO = 0.
        return FLOAT_ZERO

# Functions | Logarithm:
def sympy_logarithmic_function(
    sympy_variable_x: sp.Symbol,
    parameter_A: float,
    parameter_B: float) -> sp.Expr:
    """
    ## Description"
    Create a logarithmic function of the form A * log(B * x).
    """

    if parameter_B == 0:
        raise ValueError("parameter_B must be nonzero to avoid log(0).")
    
    try:
        return parameter_A * sp.log(parameter_B * sympy_variable_x)
    except Exception as ERROR:
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
        sympy_symbols: sp.Symbol,
        depth: int) -> sp.Function:
    """
    ## Description:
    We use SymPy to generate a random function of a single variable. We generate
    the function using the `depth` parameter that determines how many iterations of 
    function composition we perform.

    ## Arguments:
    1. `sympy_symbols` (sp.Symbol)
    2. `depth` (int)
    """
    
    # (X): List of available (unary) functions:
    functions = [
        sympy_constant_function,
        sympy_nth_degree_polynomial,
        # sympy_exponential_function,
        # sympy_logarithmic_function,
        # sympy_sine_function,
        sympy_cosine_function,
        # sympy_tangent_function
        ]
    
    # (X): Initalize a set object to store unique elements:
    used_independent_variables = set()

    # (X): Cast the SymPy tuple of symbols into a list:
    available_expressions = list(sympy_symbols)
    
    # (X): Initialize the expression tree generation:
    for _ in range(depth):

        # (X): In generating the expression tree, we choose either U(•) or B(•, •) operators:
        operation_type = np.random.choice(['unary', 'binary'])

        # (X): In U(•) operations, we need a single input:
        if operation_type == 'unary':

            # (X): Select a random expression:
            chosen_expression = np.random.choice(available_expressions)

            # (X): Compare if the expression *is or is not* a "primitive" SymPy symbol:
            if chosen_expression in sympy_symbols:

                # (X): If the selection *is* a primitive symbol, add this symbol as "used":
                used_independent_variables.add(chosen_expression)

            # (X): Choose a random function from the list `functions`:
            function_index = np.random.randint(0, len(functions))

            # (x): Ascertain the corresponding *function* from the list!
            function = functions[function_index]

            # (X.1): If the function is a nth-degree polynomial, we need to be fancy in handling it:
            if function == sympy_nth_degree_polynomial:

                # (3.1.1): Randomly choose the degree of the polynomial: (1 ≤ n ≤ 4)
                polynomial_degree = np.random.randint(2, 5)

                # (3.1.2): ...
                coefficients = np.round(np.random.uniform(-5, 5, size = polynomial_degree + 1)) 

                # (3.1.3):
                new_expression = function(chosen_expression, polynomial_degree, *coefficients)
            
            # (3.2): Otherwise, functions only come with finite and determined number of parameters:
            else:

                # (3.2.1): Obtain the number of arguments (mathspeak: parameters) required for each function:
                number_of_arguments_per_function = function.__code__.co_argcount

                # (3.2.2): Using the number of parameters, choose them randomly from the interval [-5, 5] to parametrize the function:
                function_parameters = np.round(np.random.uniform(-5, 5, size = number_of_arguments_per_function - 2))

                # (3.2.3): Obtain the result by passing in the required arguments:
                new_expression = function(chosen_expression, 1, *function_parameters)
            
            available_expressions.append(new_expression)

        # (X): In B(•, •) operations, we need two inputs:
        else:

            # (X): We find the two inputs to the B(•, •) operator:
            if len(available_expressions) >= 2:
                first_chosen_expression, second_chosen_expression = random.sample(
                    population = available_expressions,
                    k = 2)
                
            elif len(available_expressions) == 1:
                first_chosen_expression = available_expressions[0]
                second_chosen_expression = available_expressions[0]
            
            else:
                raise ValueError("No expressions available for binary operation.")
            
            # (X): Perform a set union...
            used_independent_variables.update(set([
                first_chosen_expression,
                second_chosen_expression
                ]) & set(sympy_symbols))

            # (X): Chose a random arithmetic operation between the two:
            chosen_binary_operation = np.random.choice([sp.Add, sp.Mul, sp.Pow])
            
            # (X): Actually perform the binary operation:            
            new_expression = chosen_binary_operation(first_chosen_expression, second_chosen_expression)
            
            # (X): Add the resultant expression to the list:
            available_expressions.append(new_expression)

    # (X): The last element of `available_expressions` is the most recently-updated expression:
    final_expression = available_expressions[-1]

    # (X): "set quotient" (?) to determine if the loop missed any independent variables:
    unused_variable = set(sympy_symbols) - used_independent_variables   

    # (X): If there *are* unused variables:
    if unused_variable is not None:

        # (X): Initialize iteration over unused variables...
        for unused_variable in unused_variable:

            # (X): ...and force their inclusion into the final expression:
            final_expression += unused_variable

    return final_expression

def sympy_lambdify_expression(
        sympy_variables: list[sp.Symbol] | tuple[sp.Symbol, ...],
        sympy_expression: sp.Expr):
    """
    ## Description:
    Here, we assume you have defined (i) a collection of SymPy Symbols and
    (ii) a SymPy expression composed of those symbols. Now, we simply turn
    that SymPy function of those symbols into a Python function through
    SymPy's lambdify function.

    ## Arguments:
    1. `sympy_variables` (sp.Symbol)
    2. `sympy_expression` (int)
    """
    return sp.lambdify(sympy_variables, sympy_expression, 'numpy')