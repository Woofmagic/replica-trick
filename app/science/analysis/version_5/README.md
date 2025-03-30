# Symbolic Regression Attempt #0:

## Introduction:

Here, we attempt to get PySR to learn a Sigmoid Function in a *rich* data setting (i.e. $N_{data} = 100$) sampled equidistantly along $x$ from $[-1, 1]$.

## Underlying Function:

The underlying function that generated the experimental data was:

$$y(x) = \frac{3}{1 + e^{- a_{1} (x - a_{2})}}$$

That is, it is a Sigmoid curve, and its parameters are $a_{1} = 2.5$ and $a_{2} = 0.1$.

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally distance along the $x-$axis in the interval $[-1, 1]$.

As we have now noticed, the SR algorithm does decently well when (i) the data is *rich* (i.e. there's a lot of it), (ii) when the data is *consistently sampled*, and (iii) when we tightly constrain the algorithm's search space.

### Sampling Details:

Experimental data was obtained by evaluating the underlying function for all chosen $x$ values and then sampling from $\mathcal{N}(0, 0.052)$ to simulate experimental uncertainties. Then, a stochastic component sampled from $\mathcal{N}(0, 0.09)$ was added to the real data. (More advanced systematics to come.)

## Network Architecture:

Neurons per Layer = $[32, 16, 1]$

Activation per Layer = [relu, relu, linear]

Optimizer = Adam

Number of Replicas = $100$

Epochs per Replica = $1000$

Batch Size = $32$

Validation Split Size = $20$%

Learning Rate = $0.01$

Loss Function = MSE

## Symbolic Regression:

### Function Space:

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through the following operations:

Binary operators = [addition, subtraction, multiplication, and divison]

Unary operators = [exp]

### Average Model:

Later.

### Median Model:

Later.

# Conclusion:

## Did SR get the Function?

It appeared that SR got respectably close to the underlying function, but the issues with mismatching parameter values still remains. On the other hand, the SR algorithm, when its search space is constrained in the manner we discussed earlier, manages to find the correct symbolic representation of the data even if the numerical values for the parameters are not correct. Of course, we expect that *if we constrain the parameter space in the way that we did*, the algorithm might have *no choice but* to come up with precisely the desired equation. Thus, there is the possibility for this systematic bias in our approach.

### Comparing the Average Models:

The function that best matched the underlying one from the average model was 

$$\hat{y} (x) = \frac{2.368305}{\beta_{1} + e^(\beta_{2} x)}$$

with $\beta_{1} = 0.7965663$ and $\beta_{2} = -2.5154507$.

### Comparing the Median Models:

Later