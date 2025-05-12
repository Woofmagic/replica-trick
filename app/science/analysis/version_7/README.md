# Symbolic Regression Attempt #7:

## Introduction:

We again attempt to fit a Lorentzian. This time, we increased the number of data points from $N = 100$ to $N = 125$. Our main idea is that the parameters on the Lorentzian (which will be "discovered") will become more precise.

## Underlying Function:

The underlying function that generated the experimental data was a Lorentzian centered at $a_{1} = -0.117$ and with a HWHM (half-width at half-maximum) of $a_{2} = 0.121$. That is,

$$y(x) = \frac{1}{\pi a_{2} \left(1 + \left( \frac{x - a_{1}}{a_{1}}\right)^{2} \right)}$$

## Experimental Details:

The experiment generated $N = 125$ datapoints that were not equidistantly-spaced along the $x-$axis in the interval $[-1, 1]$.

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

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, multiplication, and division with *no* unary operators.

### Average Model:

We will tabulate the whole thing later.

### Median Model:

Later.

# Conclusion:

## Did SR get the Function?

It found a Lorentzian form, but its parameters were not great.

### Average Model

At complexity $C_{9}$, we the model found a Lorentizan with the equation:

$$\hat{y}_{9}(x) = \frac{0.040526673}{x\left(x + 0.23751053\right) + 0.029533174}.$$

As before, if one expands the form of $y(x)$ offered in the introductory section, they will obtain

$$y(x) = \frac{0.385155}{x \left(x + 0.234\right) + 0.02833}.$$

We should remark that our initial prediction seemed to have panned out! What we believed was that an increase in the number of data points helped the SR model to converge on tigher parameters. Upon comparison of our the results of the previous experiment, we find that indeed the previous values of the parameters of the model were less accurate than these ones. We are interested in how this trend proceeds as we perform the limiting analysis of $N \to \infty$ in the number of datapoints. Will these parameters converge before this asymptotic limit (to within a specified degree of acceptable precision), or do we require all of the data of the function? The answer is obvious, but we still might be able to salvage a respectable technique of analysis.

### Median Model:

The median "AI representation" model was best-fit by a $C = 9$ function with the equation

$$\hat{y}_{9}(x) = \frac{0.04055725}{x\left(x + 0.23672014\right) + 0.029410124}.$$

We remark that the median model has parameter values close to the average model, as we should expect if we are increasing the number of datapoints increases.

## What have we learned?

It seems that the more datapoints we have, the better we do at the symbolic regression. However, there are still major limitations. What is increasingly evident is that, when the function space is tightly constrained, the algorithm will have *no choice but* to "discover" the form of the underlying function. This practice is abhorrent. If one has no access to the underlying function in the slightest, there is no way by which they couuld tell if an SR algorithm has achieved anything remotely close to the true function. Additionally, it is not always realistic to expect that experiments have access to over $100$ datapoints, especially for the experiments we are interested in.

In order to better understand SR before completely giving up on it, we need to find a way to hit a sweet spot where our search space is uninhibited and the number of experimental datapoints is small. In principle, we can achieve better results if we use more replicas in our initial fit, which is the "AI representation" of the experimental data. So, future experiments ought to probe larger replicas in the initial representation before running the symbolic algorithms.

Another idea is to run the SR algorithm on each pseudodata representation of the experimental data and then perform an average across them in a fancy way.