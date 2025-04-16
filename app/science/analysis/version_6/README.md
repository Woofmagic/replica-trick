# Symbolic Regression Attempt #0:

## Introduction:

Now, we attempt to fit a Lorentzian. If an SR algorithm can uncover a Lorentzian function, we will be able to discover where its true maximum is and its width, and that is often of interest in these kinds of experiments.

## Underlying Function:

The underlying function that generated the experimental data was a Lorentzian centered at $a_{1} = -0.117$ and with a HWHM (half-width at half-maximum) of $a_{2} = 0.121$. That is,

$$y(x) = \frac{1}{\pi a_{2} \left(1 + \left( \frac{x - a_{1}}{a_{1}}\right)^{2} \right)}$$

## Experimental Details:

The experiment generated $N = 100$ datapoints that were not equidistantly-spaced along the $x-$axis in the interval $[-1, 1]$. The earlier experiment revealed that one can approach an SR-discovered Lorentzian when the data are equally spaced along the independent variable. Here, we see how it can perform in a *rich* data setting but when the data are not uniform in distribution.

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

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, multiplication, and division, and also comstraining the space of unary operators to include only the $\exp$ operator. We wanted to add the exponential to see if the SR algorithm had a chance at seeing the data as a Gaussian rather than a Lorentzian, which is another (favorite) symmetric distribution. Given the constraints on the search space, it is therefore quite likely that a Lorentizan function is discovered. In essence, all the algorithm needs to do is introduce an $x^{2}$, a division operator, and optimize a few parameters. 


### Average Model:

We will tabulate the whole thing later.

### Median Model:

Later.

# Conclusion:

## Did SR get the Function?

### Average Model

At a complexity of $C = 9$, there was perhaps the closest SR fit to the data. The symbolic function at that level of complexity was:

$$\hat{y}_{9}(x) = \frac{0.04226561}{x\left(x + 0.24000752\right) + 0.030594291}.$$

While the form offered above does not appear to be Lorentzian in nature, it in fact is; if one expands the form of $y(x)$ offered in the introductory section, they will obtain

$$y(x) = \frac{0.385155}{x \left(x + 0.234\right) + 0.02833}.$$

As we can see, while the SR algorithm has indeed found the correct syntactic form of the function, it has not found the correct parameters. Additionally, we have not yet performed a rigorous analysis of the error in these fit parameters, which further contaminates this research.

Models of higher complexity begin putting polynomial expressions in the numerator, resulting in rational expressions, which are of the form $\frac{f(x)}{g(x)}$ where $f$ and $g$ are polynomial. At even higher levels of complexities, we find that the exponential function has manifested in both the numerator and denominator; there is not much insight we can provide into this observation.

### Median Model:

At a complexity of $C = 9$, we once again obtained the closest SR fit to the data. The function we obtained was

$$\hat{y}_{9}(x) = \frac{0.042094022}{x(x + 0.24038322) + 0.030526599}.$$

As usual, we should compare the equation above to the true one below:

$$y(x) = \frac{0.385155}{x \left(x + 0.234\right) + 0.02833}.$$

The analysis proceeds similarly to the one above: While the syntactic form of the expression was indeed found, its parameters do not match, and without erorr analysis, we cannot assess how confidence the algorithm was in finding those parameters.

## What have we learned?

Several themes remain consistent; when the SR search space is constrained, the algorithm is more likely to come up with the correct syntactic form of the underlying function; when the maximum complexity of the algorithm is at least as large as the complexity of the underlying function, we may be able to uncover the syntactic form of the underlying functions; the parameters of even a syntactically-correct symbolic representation of the function do not match the true function's.

It is of major importance to understand to what extent the results we have obtained are fully self-fulfilling, in the sense that by constraining the search space provides the algorithm *no choice but to* find the correct syntactic form of the underlying function. If this is indeed the case, then by no means can we claim that the algorithm "discovered" the form of the underlying functions --- the algorithm was *pre-determined* to uncover the syntactic form of it in the first place. Additionally, the algorithm is not able to fully match the parameters of the underlying function, but without rigorous error quantification, the only thing we can say is indeed that no *match* was discovered: Had we instead managed to attach errors to each of those parameters might we take a further step in answering the question of the model's accuracy. The possibility arises, then, of using SR technology to guess at symbolic forms of functions and then employ separate technology to optimize its parameters.On the other hand, in the limit of infinitely-precise data points that lie directly on top of the true value of the underlying function, we should be able to get a perfect match.

We will now take what we have observed and perform another experiment to learn more. 