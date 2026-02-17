# Scattering Parameters

**S-parameters (Scattering parameters)** describe how incident electromagnetic waves scatter (reflect or transmit) through a multi-port network and are the standard way to characterize high-frequency interconnects.

They do *not* directly describe electric field **E** or magnetic field **H** distribution in space, but **how waves entering a network are reflected and transmitted at its ports**.

They quantify:
- How much signal is reflected
- How much is transmitted
- How much couples into other ports


## Mathematical definition
Inject wave at port *j*, and measure outgoing wave at port *i*:
$$ S_{ij} = \frac{b_i}{a_j} $$
where:
- $a_j$ = incident wave at port 
- $b_i$ = reflected (outgoing) wave at port  

At port *i*:
$$
a_i = \frac{V_i + Z_0 I_i}{2\sqrt{Z_0}}, 
b_i = \frac{V_i - Z_0 I_i}{2\sqrt{Z_0}}
$$
where:
- $V_i = $ voltage at port
- $I_i = $ current at port
- $Z_0 = $ reference impedance

## How to quantify the waves?
At each port, we define two wave quanties:
- $a_i \rightarrow$ incident wave amplitude
- $b_j \rightarrow$ reflected (outgoing) wave amplitude  
These are defined in terms of voltage and current.

## What are the "parameters" in S-parameters?
For an N-port network, you get an N\*N matrix:
$$
\mathbf{S} = \begin{bmatrix}
    S_{11} & S_{12} & \dots \\
    S_{21} & S_{22} & \dots \\
    \vdots & \vdots & \ddots
\end{bmatrix}
$$
- Diagonal terms: reflection  
- Off-diagonal terms: transmission  

## Physical wave properties:
$$ S_{ij}(\omega) = |S_{ij}(\omega)| e^{j\phi_{ij}(\omega)} $$
So it describes:
1. Magnitude → how much energy is transmitted/reflected  
2. Phase → how much delay/shift occurs  