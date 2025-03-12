## Experiment E0010

## Underlying Function:

Lorentzian, centered at $b_{1}$ with spread $a_{1}$.

$$y(x) = \frac{1}{a_{1} \pi \left( 1 + \left( \frac{x - b_{1}}{a_{1}} \right)^{2} \right)}$$

### Parameters:

$a_{1} = 0.121$, $b_{1} = -0.117$.

### Data Sparsity:

Sparse! 

$N = 30$.

## Replica Setup:

Number of Replicas: 20/50 (failure in multithreading)

Epochs per Replica: 2000

## Network Architecture:

Neurons per Layer: [480, 320, 240, 120, 32, 16, 1]

Activation Functions: [sigmoid, sigmoid, sigmoid, sigmoid, sigmoid, sigmoid linear]

Loss: MSE

Optimizer: Adam