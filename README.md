# replica-trick
Exploring the Replica Trick as applied to DNNs.

It turned out that a lot of these other repositories can all be put together.

## Some Definitions/Information:

replica data sets: 

"These sets are obtained from original data by generating random artificial data points according to Gaussian probability distribution with a width defined by the error bar of experimental measurements."

## Point:

The purpose of this program is to attempt to learn an underlying one-dimensional function $f(x)$ with a finite set of data points using neural networks.

Every time the program is run, a different underlying function is generated using the SymPy library. We call that function a "law of nature" or something pretentious like that. Then, any experiment that we do involves a measurement of $f(x)$ with some experimental error. The "experimental data" is a set of points, $\{ \left(x_{i}, y_{i} \right)\}_{i = 1}^{N}$.