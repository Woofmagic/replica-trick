## Experiment E0012

## Underlying Function:

Sigmoid.

$$y(x) = \frac{3}{1 + e^{-a_{1} \left( x - b_{1}\right)}}$$

### Parameters:

$a_{1} = 2.5$, $b_{1} = 0.1$. (Stupid of me, but that $3$ in the numerator is magic, and should really be absorbed into another parameter.)

### Data Sparsity:

$N = 76$.

## Replica Setup:

Number of Replicas: 19/50 (failure in multithreading)

Epochs per Replica: 2000

## Network Architecture:

Neurons per Layer: [480, 320, 240, 120, 32, 16, 1]

Activation Functions: [sigmoid, sigmoid, sigmoid, sigmoid, sigmoid, sigmoid linear]

Loss: MSE

Optimizer: Adam