# Symbolic Regression Attempt #0:

## Introduction:

We are just trying out the workflow for now. As you can tell, there are still multiple programmatic improvements to be made. The most important part of this trial run, though, was the information that the network architecture and training procedure needs to be rethought in a major way. For such a deep neural network, it was unable to fit a simple line.

## Underlying Function:

The underlying function that generated the experimental data was:

$$y(x) = 0.65 x - 0.18$$

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally distance along the $x-$axis in the interval $[-1, 1]$. We thought that by providing this much information to the neural network in such a consistent manner that it would assist not only the training of the network but of the symbolic regression analysis as well.

### Sampling Details:

Experimental data was obtained by evaluating the underlying function for all chosen $x$ values and then sampling from $\mathcal{N}(0, 0.052)$ to simulate experimental uncertainties. Then, a stochastic component sampled from $\mathcal{N}(0, 0.09)$ was added to the real data. (More advanced systematics to come.)

## Network Architecture:

Neurons per Layer = $[32, 16, 1, ]$

Activation per Layer = [relu, relu, linear]

Optimizer = Adam

Number of Replicas = $2$

Epochs per Replica = $1000$

Batch Size = $32$

Validation Split Size = $20$%

Learning Rate = $0.005$

Loss Function = MSE

## Symbolic Regression:

### Function Space:

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, and multiplication with *no* unary operators to explore. The real numbers equipped with addition and subtraction is a Group, but then

### Average Model:

Later.

### Median Model:

Later.

# Conclusion:

## Did SR get the Function?

Later