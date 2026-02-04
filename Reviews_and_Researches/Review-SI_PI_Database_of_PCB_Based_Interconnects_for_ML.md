# Review of SI/PI Database Paperss

A brief review of paper [**“SI/PI-Database of PCB-Based Interconnects for Machine Learning Applications”**](../Documents/SI_PI-Database_of_PCB-Based_Interconnects_for_Machine_Learning_Applications.pdf).  
Paper Website: <https://ieeexplore.ieee.org/document/9361755>


**Paper summary**:  
The setup of the database, its data sets, and examples on how to apply ML techniques to the data.
- Introduces a database of PCB-based interconnect simulations for ML research;
- Proves that ML can accelerate PBC interconnect analysis;


**Domains**:
- SI (Signal Integrity)
- PI (Power Integrity)
- EMC (Electronic Compatibility)

## PB(Physics-Based) vs ML(Machine Learning)

PB(Physics-Based) approaches:
- ✔ Physically accurate
- ✔ Generalizable (new geometries still valid)
- ✖ Computationally expensive
- ✖ Slow for large design sweeps

ML approaches
- ✔ Very fast once trained
- ✔ Good for optimization and exploration
- ✖ Need large datasets
- ✖ Limited extrapolation beyond training data

Key idea of the paper:
- Use PB simulations to generate “ground-truth” data
- Then train ML models to approximate PB results at much lower cost

## Terminology Notes
- PCB: Printed Circuit Board 
- SI: Signal Integrity
- PI: Power Integrity
- EMC: Electronic Compatibility
- PDN: Power Delivery Network
- PB: Physics-Based  
    > A Physics-Based (PB) model or tool is one that explicitly solves Maxwell’s equations (or reduced physical equations derived from them) to predict electromagnetic behavior.  
- ANNs: artificial neural networks
    > ANNs (Artificial Neural Networks) are data-driven models that learn a nonlinear mapping from inputs to outputs by stacking simple computational units (“neurons”) and adjusting their parameters from data.  
- via:
    > A via is a vertical electrical connection that passes through one or more layers of a printed circuit board (PCB).  