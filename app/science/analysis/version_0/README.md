# Symbolic Regression Attempt #0:

## Introduction:

We are just trying out the workflow for now. As you can tell, there are still multiple programmatic improvements to be made. The most important part of this trial run, though, was the information that the network architecture and training procedure needs to be rethought in a major way. For such a deep neural network, it was unable to fit a simple line. (NOTICE: There was a major issue preventing the program from actually finding the correct form of the data. In essence, then, the rest of the content below should be considered devoid of scientific analysis/)

## Underlying Function:

The underlying function that generated the experimental data was:

$$y(x) = 0.65 x - 0.18$$

## Experimental Details:

The experiment generated $N = 100$ datapoints that were spaced equally distance along the $x-$axis in the interval $[-1, 1]$. We thought that by providing this much information to the neural network in this consistent manner that it would assist not only the training of the network but of the symbolic regression analysis as well.

### Sampling Details:

Experimental data was obtained by evaluating the underlying function for all chosen $x$ values and then sampling from $\mathcal{N}(0, 0.052)$ to simulate experimental uncertainties. Then, a stochastic component sampled from $\mathcal{N}(0, 0.09)$ was added to the real data. (More advanced systematics to come.)

## Network Architecture:

Neurons per Layer = $[32, 16, 1, ]$

Activation per Layer = [relu, relu, linear]

Optimizer = Adam

Number of Replicas = $20$

Epochs per Replica = $2000$

Batch Size = 32

Loss Function = MSE

## Symbolic Regression:

### Function Space:

We allowed for a *full exploration* of the known binary operators and unary operators. That is, we allowed searching through addition, subtraction, multiplication, and division, and we also allowed exponentials, logarithms, and trigonometric functions. In retrospect, such an approach made it difficult for the SR to settle on a decent and simple representation of the underlying data. 

### Average Model:

The model that most closely matched the underlying function was simply the first choice from the first complexity class $(C = 1)$, which waas:

$$\hat{y}_{1}(x) = x.$$

We should notice immediately that the selected function misses out on two key parameters in a linear regression: the slope and the intercept. (Technically, the slope in the given equation is $1$.) What is remarkable is that the algorithm was unable to detect the shift in the slope and the y-intercept of the data despite the data being evenly distributed across $x$ and relatively close to the actual underlying function from which it was generated. Is it the case that these SR algorithms are not particularly attentive to small values of these parameters which can only manifest when traced through a larger domain of paramter space? Is there something wrong with the parameter optimization part of the SR? It is hard to imagine that these questions were not considered before. We will follow-up on these ideas in future experiments.

### Median Model:

The median model similarly discovered a simple line with a slope of $1$ and a y-intercept of $0$.

$$\hat{y}_{1}(x) = x.$$

# Conclusion:

## Did SR get the Function?

In this first experiment, the SR algorithm did not uncover the underlying function. It is remarkable that it did not, because this should indeed be the first thing that one tries to fit using these types of algorithms. So, we will have to immediately follow-up on debugging the code and learning more about the algorithm.