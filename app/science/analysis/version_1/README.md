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

We *did not* explore the full space of functions $f : \mathbb{R} \to \mathbb{R}$ in this analysis. We only explore those $f$s that were reachable through addition, subtraction, and multiplication with *no* unary operators to explore. In this way, we *severely* restrict the full space of functions reachable through these operations. It is from this angle that we ought to expect the SR algorithm to settle on the correct expression in the end. The entire space of polynomials is in principle reachable if we only allow a single variable $x$ to be added, subtracted, and multiplied by constants or itself. We should thus *expect* that the algorithm uncovers the form of a line, but perhaps yields incorrect parameter values. We should also expect that, given we are only using $N = 2$ replicas, that there will be some uncertainty in the SR values of these parameters.

### Average Model:

Our SR algorithm did indeed find a line in the seventh complexity class $(C = 7)$:

$$\hat{y}_{7} = 0.6576989x - 0.182799999999986.$$

We compare the equation above with the true one of

$$y(x) = 0.65 x - 0.18.$$

We notice that the algorithm has found the correct parameters to within about 2%, which is impressive. On the other hand, the data follows a line. 

The next-highest complexity class offers a quadratic function. The one beyond that offers a cubic function, and the one after that offers a quntic polynomial. 

### Median Model:

As we expected, the median model also came up with a line in the seventh complexity class:

$$\hat{y}_{7} = 0.6587867x - 0.18269999999386.$$

# Conclusion:

## Did SR get the Function?

We can say that the SR algorithm did indeed uncover the underlying function of a simple line. Such a result indicates that we know that there is some hope in further understanding the libraries and code that we are using to rigorously explore SR. In other words, this experiment was nothing but a test of the software. Nevertheless, there are some interesting avenues that we might pursue.

We imagine that, because we are fitting a line of data, it is quite trivial for such a (highly-constrained) SR algorithm to spit out the correct syntax of the underlying function and even recover a few of its parameters. What, then, is the smallest number of replicas that we need in order for the SR algorithm to settle on a simple line? We might need only a single replica. Indeed, there remains a question about methodology: Why *not* just do an SR fit given the data in the first place? Why do we need to perform a replica average *first* and *then* do a symbolic regression?

Additionally, there is an issue of "boundedness": If not constrainted to a maximum depth, the algorithm will continue to suggest more and more complex functions that "beautifully fit" the data --- These are functions that overfit. Therefore, in order to settle on a "right answer" for the SR process, we must be able to somehow know within what complexity range the true solution lies. Otherwise, we have no justifiable means to proclaim that the seventh complexity class is the true answer in a truly blinded scenario. Indeed, there are SR solutions of higher complexity class that better minimize the loss.