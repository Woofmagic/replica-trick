# Symbolic Regression Attempt #3:

## Introduction:

We now attempt to throw SR (with a constrained search space) at more advanced functions. In this experiment, we will see if we can fit a Lorentzian. Lorentzian functions are important in the study of resonant systems, and it is of critical interest to see if, with a given SR fit, we can find the center (peak) and the spread of the function.

## Underlying Function:

The underlying function that generated the experimental data was a Lorentzian centered at $a_{1} = -0.117$ and with a HWHM (half-width at half-maximum) of $a_{2} = 0.121$. That is,

$$y(x) = \frac{1}{\pi a_{2} \left(1 + \left( \frac{x - a_{1}}{a_{1}}\right)^{2} \right)}$$

We will note now that this function may be equivalently expressed according to

$$y(x) = \frac{0.0385155}{x^{2} + 0.234 x + 0.02833}.$$

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally along the $x-$axis in the interval $[-1, 1]$. This experimental configuration is how we are usually studying SR because it provides a robust scaffolding for the algorithms so that they can perform their best. We will then plan to gradually strip away more and more features of this configuration and measure how the algorithms change then. 

### Sampling Details:

Experimental data was obtained by evaluating the underlying function for all $100$ $x$ values and then sampling from $\mathcal{N}(0, 0.052)$ to simulate experimental uncertainties. Then, a stochastic component sampled from $\mathcal{N}(0, 0.09)$ was added to the real data. (More advanced systematics to come.)

## Network Architecture:

Neurons per Layer = $[32, 16, 1]$

Activation per Layer = [relu, relu, linear]

Optimizer = Adam

Number of Replicas = $100$

Epochs per Replica = $1000$

Batch Size = $32$

Validation Split Size = $20$%

Learning Rate = $0.005$

Loss Function = MSE

## Symbolic Regression:

### Function Space:

We constrained our search space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, multiplication, and division. These are the only operations that we need in order to "discover" a Lorentzian. Notice how multiplication effectively covers the unary operator of squaring a number; in this way, we effectively reach rational expressions. (It has been noticed before that keeping the "pow" unary operator in the search algorithm can often blow up possible solutions to the problem.)

### Average Model:

Our SR algorithm found a few strange representations of a Lorentzian curve. There were actually three representations of the Lorentzian that were found. Specifically, models $\hat{y}_{11}$ and $\hat{y}_{13}$ found a Lorentzian plus and minus (respectively) a constant y-offset. (As usual, the notation $\hat{y}_{i}$ refers to the model's ($\hat{y}$) fit of the $i-$th complexity class.) So, technically, the "more correct" fit is actually the SR represetation of a pure Lorentzian curve without an offset, which was:

$$\hat{y}_{9} = \frac{0.039477196}{x(x + 0.2299231) + 0.028298885} = \frac{0.0394772}{x^{2} + 0.229923x + 0.0282989}.$$

Compare the SR fit above to the true, underlying function below:

$$y(x) =\frac{0.0385155}{x^{2} + 0.234 x + 0.02833}.$$

Technically, $\hat{y}_{9}$ is the "most correct" representation because it does not feature any sort of y-offset like the other two do. With that in mind, we notice that the model parameters are within about 1-2% as before.

### Median Model:

We expect that the median replica model will also come up with similar solutions for the fit. And indeed it does:

$$\hat{y}_{9} = \frac{0.039413266}{x(x + 0.22966512) + 0.028222235} = \frac{0.0394133}{x^{2} + 0.229665x + 0.0282222}.$$

# Conclusion:

## Did SR get the Function?

We can say that the SR algorithm did indeed "discover" a Lorentzian curve. Such a "discovery" may be useful in the future if we want to predict the resonant frequency of a particular bulk material, for example: The PySR algorithm produced a representation of the Lorentzian which then enables its analysis. Of course, the parameters on the SR fit are not precisely those of the underlying function, they are within 1-2% as discussed above.

Despite this discussion, we must keep in mind the major limitations and constraints of the SR that we just ran, which include the effective coercian of the "right answer" into the algorithm by imposing various limitations on the search space.