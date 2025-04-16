# Symbolic Regression Attempt #2:

## Introduction:

We throw our SR infrastructure at a quadratic equation. Given that we have learned that polynomials are easily selected for when limiting our space of possible functions, we want to verify that our SR algorithm, with more replicas, will be able to fit a simple quadratic and possibly better estimate its parameters. This experiment represents the testing of that hypothesis.

## Underlying Function:

The underlying function that generated the experimental data was:

$$y(x) = 1.02 x^{2} - 2.78 x + 3.4.$$

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally distance along the $x-$axis in the interval $[-1, 1]$. We thought that by providing this much information to the neural network in such a consistent manner that it would assist not only the training of the network but of the symbolic regression analysis as well.

### Sampling Details:

Experimental data was obtained by evaluating the underlying function for all chosen $x$ values and then sampling from $\mathcal{N}(0, 0.052)$ to simulate experimental uncertainties. Then, a stochastic component sampled from $\mathcal{N}(0, 0.09)$ was added to the real data. (More advanced systematics to come.)

## Network Architecture:

Neurons per Layer = $[32, 16, 1]$

Activation per Layer = [relu, relu, linear]

Optimizer = Adam

Number of Replicas = $50$

Epochs per Replica = $1000$

Batch Size = $32$

Validation Split Size = $20$%

Learning Rate = $0.005$

Loss Function = MSE

## Symbolic Regression:

### Function Space:

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, and multiplication with *no* unary operators to explore. As before, we expect to be able to reach polynomials of some degree $n$ with this limitation, and thus we predict that our SR algorithm will indeed find *a* quadratic function; the question remains of how accurate its parameters will be in the end.

### Average Model:

Our SR algorithm did indeed find a quadratic equation in the seventh complexity class $(C = 7)$, but this equation is actually not the best fit. Instead, the equation in complexity class $9$ fit the function better --- quantitatively verifiable with the loss $L$, and qualitatively verifiable by comparing its coefficients. For reference, here are the two equations that are quadratic and thus serve as a decent fit to the underlying function:

$$\hat{y}_{7} = x^{2} - 2.7570803x - 3.14142783$$

and 

$$\hat{y}_{9}(x) = 1.0117995x(x - 2.7246938) + 3.4103463 = 1.0118 x^{2} - 2.75684 x + 3.4103463.$$

We compare the equation above with the true one of

$$y(x) = 1.02 x^{2} - 2.78 x + 3.4.$$

The algorithm has found the correct symbolic representation of the quadratic equation with parameters that are within about 2-3% of the actual values. Again, it is impressive that the parameters are found to within this percent error, but our initial hypothesis was correct, and paints this "finding" in a different light --- given the highly-constrained form of the function space, it was effectively *inevitable* that the SR algorithm would settle on a quadratic function and *then* optimize its parameters.

### Median Model:

We expect virtually the same findings as above in the case of the median model. We find, again, two quadratic functions in the output of PySR, and they differ only in their coefficients, thus qualifying both for a possible fit. The solution in complexity class $9$, again, minimized the loss $L$ better than that on complexity class $7$, and thus it should serve as the effective "final guess" of the underlying function. For reference, the two functions that the SR algorithm found are:

$$\hat{y}_{7} = 0.6587867x - 0.18269999999386$$

and 

$$\hat{y}_{9} = 1.0117995 x (x - 2.7246938) + 3.4103463 = 1.0117995 x^{2} - 2.75684 x + 3.4103463.$$

# Conclusion:

## Did SR get the Function?

The SR algorithm did indeed find the correct symbolic form of the function, but this was to be expected: With such density of data distributed uniformly in the parameter space and with minimal experimental error *and* with the constraints on the SR algorithm, there was little chance for it to *not* find the correct function.

Since we are now confident in some of the mechanics of this SR algorithm, let us attempt to fit more complicated functions with the same "density" of data and constraints on the search space.