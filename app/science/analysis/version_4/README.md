# Symbolic Regression Attempt #4:

## Introduction:

We now turn our attention to the Gaussian function. Can our SR approach recover a Gaussian function?

## Underlying Function:

The underlying function that generated the experimental data was a Gaussian with a mean and variance that is retrospectively hard to determine because the function was not written in its "standard form." See the equation below for the from of the underlying function from which the experimental data was sampled:

$$y(x) = \frac{1}{a_{2} \sqrt{2 \pi} e^{- \frac{2}{2} \left( \frac{x - a_{1}}{a_{2}} \right)^{2}}}.$$

Note that the "standard form" of the Gaussian includes a $1/2$ in the exponent, but this was overlooked in the code. In light of this oversight, we will now remark that the function may be equivalently expressed according to

$$y(x) = \frac{3.3042372952642 e^{-21.8359682068303\left(x - 0.145\right)^{2}}}{\sqrt{\pi}}$$

or, even further,

$$y(x) = 1.8642162635581 e^{-21.8359682068303\left(x - 0.145\right)^{2}}.$$

We should remark at the outset that there is already a major issue in the representation of this function. The symbolic regression results will likely find an "exponential-like" function that we will have to symbolically coerce into an identifiable form. Glancing at the expression above, one can immediately glean at least a dozen more ways to represent it, and some of those involve expanding the square, rearranging terms in the exponential, but also writing the numerical prefactor as an exponentiated constant as well, then perhaps recombining it into the square. This "degreneracy of representations" is a major problem in SR (I think): For how many different (unique!) ways are there to write $y(x) = d e^{ax^{2} + bx + c}$?

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally along the $x-$axis in the interval $[-1, 1]$. We also used $N = 100$ replicas.

### Sampling Details

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

We constrained our search space of functions $f : \mathbb{R} \to \mathbb{R}$ to include only addition, subtraction, multiplication, and division as binary operators, and only included the exponential function in our set of unary operators. Therefore, as previous experiments have indicated, it is more than likely that the SR will end up "discovering" a Gaussian function.

### Average Model:

The SR-fit average replica model definitiely discovered a Gaussian, but we still need to determine if it is in the same "equivalence class" as the correct underlying function.

### Median Model:

See above.

# Conclusion:

## Did SR get the Function?

Later.