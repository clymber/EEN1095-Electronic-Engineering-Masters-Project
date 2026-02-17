# Overview of _Demystifying Machine Learning for Signal and Power Integrity Problems in Packaging_

## Preditive models with NNs
Surrogate modeling.

### Feedforward Neural Networks (FFNN)
Behavioral modeling of:
- VCOs
- Embedded passives
- Eye diagrams
- PDN impedance

Basic neuron: $y = \sigma(Wx + b) $
W: weights
b: bias
σ: nonlinear activation

### Recurrent Neural Networks (RNN)
Output depends on past time history: Input–output driver modeling.

$$
    i(t) = f(v(t), v(t-h), ..., i(t-h), ...)
$$

RNN captures memory:
- Suitable for time-domain SI analysis.
- Can reproduce eye diagrams.
- Achieves ~300× simulation speedup.

### Convolutional Neural Networks (CNN)
- Broadband S-parameter prediction
- Frequency response modeling
- High-dimensional outputs

Key insight:
> Frequency responses are spatially correlated → convolutional layers reduce parameter explosion.

### Physics-Enforced Neural Networks
Pure ML can violate physics (e.g., passivity, causality).

Solution:  
1. Add constraint layers:
- Causality Enforcement Layer (CEL)
- Passivity Enforcement Layer (PEL)

2. goals:
- S-parameters remain passive
- Physical realizability is guaranteed

