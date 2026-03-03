# Explore Possible ML Formulations for my MSc Project

## Model inputs and outputs
### Inputs (X)  
The model inputs are parameters come from `parameter.csv` file in the dataset.  

#### A description of table columns:  
| Column       | Physical Meaning | Typical Unit | What It Controls in SI Terms | Expected Impact on S-Parameters |
|--------------|------------------|--------------|-------------------------------|----------------------------------|
| EPS          | Relative permittivity (εr) of dielectric material | – (dimensionless) | Electric field propagation speed | Higher EPS → slower wave velocity, more delay, potential impedance shift |
| TAND         | Loss tangent (tanδ) of dielectric | – (dimensionless) | Dielectric loss mechanism | Higher TAND → higher insertion loss, especially at high frequency |
| PITCH        | Spacing between adjacent vias in via array | mm (typically) | Mutual coupling between vias | Larger PITCH → reduced coupling, often lower crosstalk and improved IL |
| TRACE_LEN    | Length of transmission line segment | mm | Conductor loss + phase delay | Longer TRACE_LEN → higher insertion loss, larger phase shift |
| VIAR         | Via radius | mm | Via inductance & capacitance | Larger VIAR → lower inductance but higher capacitance (impedance shift) |
| ANTIPADR     | Antipad radius (clearance hole in reference plane) | mm | Via-to-plane capacitance | Larger ANTIPADR → reduced parasitic capacitance |
| TDIEL        | Dielectric thickness between signal and reference plane | mm | Characteristic impedance control | Larger TDIEL → higher impedance (for fixed width) |
| DISTTL       | Distance between adjacent transmission lines | mm | Crosstalk strength | Larger DISTTL → lower near-end/far-end crosstalk |
| TLWIDTH      | Transmission line width | mm | Characteristic impedance | Larger TLWIDTH → lower impedance |
| SIMU_INDEX   | Simulation identifier (maps to simu_XXXX.s12p file) | – | Dataset indexing only | Used to locate corresponding S-parameter file |

#### Sample vector design
The vector for simulation $k$ (ML input) is:
$$
\mathbf{x}_k = [\text{EPS}, \text{TAND}, \text{PITCH}, \dots, \text{TLWIDTH}]
$$


### Outputs (Y) 
Targets / Labels are derived from `.sNp` files, but not directly. They come from post-processed features extracted from `.sNp` files, for examples:
- S21 real/imag
- |S21| dB
- IL
- Sdd21 (after mixed-mode conversion)
- Worst-band IL
- Group delay

**The targets / labels**
For a `.sNp` file:
$$
S_{ij}(f) for i, j = 1...12
$$
So the raw label is a 3D tensor (#FrequencyPoints, #Ports, #Ports):
$$
\mathbf{Y}_k = \{ S_{ij}(f_m) \}
$$

## Pipeline and formulations design:

### Pipeline design
```
[ parameter.csv ]
        ↓
[ design vector `X` ]
        ↓
[ Neural Network ]
        ↓
[ predicted S-parameters ]
        ↓
[ Compare with processed labels extracted from `.sNp` ]
```

### ML formulations
Possible formulations:
1. Scalar regression
2. Multi-output regression (full frequency curve per link)
3. Full S-matrix prediction

The following designs adapt IL (Insertion Loss) as metric. There may be other better options.

#### Formulation 1 - Scalar regression
Predicts   
$$
    Y_k = IL_{mean}(10\,\text{GHz})
$$
- \[y\] Easy, may be a good first baseline
- \[y\] Correlation insight
- \[x\] Waste most frequency information

#### Formulation 2 - Multi-output regression (full frequency curve per link)

$$
    Y_k = [ IL(f_1), IL(f_2), \dots, IL(f_{200}) ]
$$
- \[y\] Predicts full frequency response
- \[x\] Output dimension = 200 × 6 links = 1200 outputs.


#### Formulation 3 - Full S-matrix prediction
Predicts the entire S-matrix tensor:
$$
    S_{ij}(f)
$$
- \[y\] Most complete
- \[x\] 200 × 12 × 12 = 28,800 outputs per sample
- \[x\] Harder to train

### What is a touchstone file  
A `.sNp` is a Touchstone file that stores the scattering parameters (S-parameters) of an N-port network over frequency, where N is the number of ports. It is the standard file format used in RF(Ratio Frequency), microwave, and SI/PI engineering.

**What does a touchstone file `.sNp` file contain?**  
Suppose $N = 12$ and the number of frequency points is 200.  
A touchstone file contain multiple (e.g. 200) frequency points, and at each frequency point followed by a S-parameter matrix: $\mathbf{S}(f) \in \mathbb{C}^{12 \times 12}$.  
So the file contains: $ (200 × 12 × 12) $ complex numbers, and each complex number represents magnitude and phase of coupling between two ports.

**File structure (Touchstone format)**  
A typical header:
```
# GHz S RI R 50
```
- Frequency unit: GHz
- Parameter type: S
- Data format: RI (Real / Imag)
- Reference impedance: 50 Ω  
Then each row contains:
```
frequency  S11_re S11_im  S21_re S21_im  ... SNN_re SNN_im
```

### What are S-parameters?  
S-parameters describe how electromagnetic waves propagate through a network.  
For an N-port system:
$$ \mathbf{b} = \mathbf{S} \mathbf{a} $$
Where:  
- $a$ = incident waves  
- $b$ = reflected/transmitted waves  
- $S$ = scattering matrix  
- Each element: $ S_{ij}(f) $ means: Ratio of outgoing wave at port $i$ due to an incoming wave at port $j$, at frequency $f$.

Note-taking: [Scattering_Parameters.md](./Scattering_Parameters.md)

