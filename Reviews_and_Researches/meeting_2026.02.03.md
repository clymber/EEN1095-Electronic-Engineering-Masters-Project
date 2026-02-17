# Use Neural Networks to Predict Scattering Parameters
In circuits, the input and output voltages in a two-port network are related by Z or Y parameters. However, at high frequencies, scattering parameters are often employed as they are easier to measure.
$$
    \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
    = Z \begin{bmatrix} i_1 \\ i_2 \end{bmatrix}
$$
$$

    \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
    =
    S \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}
$$
Where:
- $v_1, v_2$ are voltages and $i_1, i_2$ are currents.
- $b_1, b_2$ and $a_1, a_2$ are forward and backword travelling waves.  
They are useful for determining metrics that give an indication of cross-talk between interconnects and distortion of signals.

Given a set of parameters (dimensions of the interconnects), train a Neural Network model to predict the scattering parameters. The predictions must physically consistent. For example, insertion loss is a metric that is used. This metric measures the loss of power in a signal when it travels through a system. It is a positive quantity. So the neural network must be such that it produces a positive quantity. This feature was investigated in:
[Chen, L., & Tan, L. (2021). __*Physics-Enforced Modeling for Insertion Loss of Transmission Lines by Deep Neural Networks*__. 2021 IEEE Asia-Pacific Microwave Conference (APM.C), 276-278.](https://doi.org/10.1109/APMC52720.2021.9661782)

The type of neural network must be considered.
A convolutional network is recommended in the following:
[Swaminathan, M., Torun, H.M., Yu, H., Hejase, 1. A., & Becker, W. D. (2020).
__*Demystifying Machine Learning for Signal and Power Integrity Problems in Packaging*__. IEEE Transactions on Components, Packaging, and Manufacturing Technology (2011), 10(8), 1-1.](https://doi.org/10.1109/TCPMT.2020.3011910)

Available datasets are described in:
[Schierholz, M., Sanchez-Masis, A., Carmona-Cruz, A., Duan, X., Roy, K., Yang, C., Rimolo-Donadio, R., & Schuster, C. (2021). SI/PI-Database of PCB-Ba:
sed Interconnects for Machine Learning Applications. IEEE Access, 9, 34423-34432.](https://doi.org/10.1109/ACCESS.2021.3061788)


Suggested reading:
[Hong, C., Li, Z., Guan, W. et al. __*Signal integrity research of high-speed interconnection systems based on scattering parameters*__. Sci Rep 15, 23096 (2025).](https://doi.org/10.1038/s41598-025-08567-1)

Challenges
- size of datasets
- Formation of physically consistent models
- The type of neural network
