## Experiment E0000

## Underlying Function:

$$y(x) = x.$$

### Parameters:

Technically, we can say $y(x) = 1 \times x$, and so the one parameter is $a = 1$.

## Replica Setup:

Number of Replicas: 50

Epochs per Replica: 2000

## Network Architecture:

Neurons per Layer: [256, 256, 256, 256, 256, 1]

Activation Functions: [sigmoid, sigmoid, sigmoid, sigmoid, sigmoid, linear]

Loss: MSE

Optimizer: Adam