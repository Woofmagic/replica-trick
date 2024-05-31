from numpy.random import normal

def sample_from_numpy_normal_distribution(
        mean_value: float,
        standard_deviation: float) -> float:
    """
    # Description
    --------------
    Uses Numpy's function random.normal() to sample a normal distribution
    with a mean of "loc" and a standard deviation of "scale," the two WORST
    KWARGS I'VE EVER SEEN IN PYTHON.

    
    # Parameters
    --------------
    mean : (float)

    standard_deviation : (float)

    # Returns
    --------------
    randomly_sampled_variable : (float)

    # Function Flow
    --------------
    (1): Sample a Normal Distribution with mean mean_value and
        standard deviation standard_deviation. If it doesn't work,
        return None.

    
    Notes
    --------------
    (1): The quadratic function has three "parameters." These are roughly
        called "A", "B", and "C." Only after specifying those three parameters
        will the function actually return an output given an input. In other
        words, besides supplying an independent variable, you need those
        three other pieces of information -- the parameters -- in order to
        get a number as an ouput.
    """
    try:
        randomly_sampled_variable = normal(
            loc = mean_value,
            scale = standard_deviation
        )
        return randomly_sampled_variable
    
    except Exception as E:
        print(f"> Error in sampling normal distribution:\n> {E}!")
        return 0.