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

Neurons per Layer = $[64, 128, 128, 64, 32, 1]$

Activation per Layer = [relu, relu, relu, relu, relu, linear]

Optimizer = Adam

Number of Replicas = $100$

Epochs per Replica = $2000$

Batch Size = 32

Loss Function = MSE

## Symbolic Regression:

### Average Model:

1. $(C = 1)$: $y(x) = -8.731149\times 10^{-11}$ (loss: $7.6232965 \times 10^{-21}$)
2. $(C = 2)$: $y(x) = \tan \left( 0 \right)$ (loss: $0$)

### Median Model:

1. $(C = 1)$: $y(x) = 1476.4445$ (loss: $8.9609096\times 10^{7}$)
2. $(C = 3)$: $y(x) = 28743.824 ^ x$ (loss: $7.562787\times 10^{7}$)
3. $(C = 4)$: $y(x) = cos(x) ^ -16.678219$ (loss: $7.391058\times 10^{7}$)
4. $(C = 6)$: $y(x) = cos(x + 0.015458494) ^ -16.868996$ (loss: $7.112467\times 10^{7}$)
5. $(C = 7)$: $y(x) = (x - -1.4218528) ^ (x * 11.8652725)$ (loss: $6.6249532\times 10^{7}$)
6. $(C = 8)$: $y(x) = tan(x + -0.570824) + (28744.186 ^ x)$ (loss: $6.2641404\times 10^{7}$)
7. $(C = 9)$: $y(x) = (x - -1.0160303) ^ ((x * 8.699294) - -6.160511)$ (loss: $6.2599828\times 10^{7}$)
8. $(C = 10)$: $y(x) = x * ((30568.5 ^ x) - tan(-0.570824 + x))$ (loss: $6.2573676\times 10^{7}$)
9. $(C = 11)$: $y(x) = exp(exp(exp(cos((0.11961924 ^ x) * 1.5018098))) + -4.87842)$ (loss: $6.1387508\times 10^{7}$)
10. $(C = 12)$: $y(x) = (((x - -1.4218519) ^ 11.8652725) ^ x) + tan(x - 0.79082835)$ (loss: $5.6507616\times 10^{7}$)
11. $(C = 13)$: $y(x) = (tan(x - 0.79082835) + tan(x + -0.570824)) + (28744.186 ^ x)$ (loss: $5.285625\times 10^{7}$)
12. $(C = 14)$: $y(x) = tan(x - 0.79082835) + ((x - -1.0160303) ^ ((x * 8.699294) - -6.160511))$ (loss: $5.2819404\times 10^{7}$)
13. $(C = 15)$: $y(x) = tan(x - 0.79082835) + ((x * (30568.5 ^ x)) + tan(-0.570824 + x))$ (loss: $5.2788772\times 10^{7}$)
14. $(C = 16)$: $y(x) = tan(x - 0.79082835) + exp(exp(exp(cos((0.11961924 ^ x) * 1.5018098))) + -4.87842)$ (loss: $5.1596104\times 10^{7}$)
15. $(C = 18)$: $y(x) = (tan(x - 0.79082745) + exp(exp(exp(cos((0.11961924 ^ x) * 1.5018098))) + -4.787197)) + -1260.2744$ (loss: $5.035127\times 10^{7}$)
16. $(C = 20)$: $y(x) = (tan(x - 0.79082835) + tan((x * x) - -0.99307984)) + ((28217.463 ^ x) + tan(x + -0.570824))$ (loss: $4.8147908\times 10^{7}$)
17. $(C = 21)$: $y(x) = (tan(x - 0.79082835) + ((x - -1.0177071) ^ ((x * 8.723608) + 6.122214))) + tan((x * x) - -0.99307984)$ (loss: $4.7958184\times 10^{7}$)
18. $(C = 22)$: $y(x) = (tan(x - 0.79082835) + tan((x * (0.00011751706 + x)) - -0.99307984)) + ((28217.463 ^ x) + tan(x + -0.570824))$ (loss: $4.635641\times 10^{7}$)
19. $(C = 23)$: $y(x) = tan(x - 0.79082835) + (tan((x * (0.00011751706 + x)) - -0.99307984) + ((x - -1.0177071) ^ ((x * 8.723608) + 6.122214)))$ (loss: $4.567347\times 10^{7}$)
20. $(C = 25)$: $y(x) = tan(x - 0.79082835) + (((x - -1.0177071) ^ ((x * 8.723608) + 6.122214)) + (tan((x * (x + 0.00011751706)) - -0.99307984) / 1.4697701))$ (loss: $4.4979036\times 10^{7}$)

# Conclusion:

## Did SR get the Function?

No. SR did not get the underlying function of $y(x) = 0.65 x - 0.18$.