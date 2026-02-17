# A brief review of CircuitNet project

This is a database project that aims to collect EDA(Electronic Design Automation) task-specific datasets for ML, similar with ImageNet project. Unlike ImageNet project, which works for computer vision, CircuitNet project is for chip design.

## Project resources
- Repository: https://github.com/circuitnet/CircuitNet
- Homepage: https://circuitnet.github.io

## EDA problems
Classic EDA tasks:  
- placement, routing, congestion, timing, power, etc.

EDA problems:
- Highly structured (graph + geometry),
- Hard to simulate,
- Usually locked inside commercial tools.

## CircuitNet Structure
### Dataset
Provides realistic circuit design samples derived from standard design flows. These samples encode:
- Netlists (connectivity graphs),
- Cell placements,
- Routing and congestion maps,
- Timing and power features,
- Technology-dependent attributes.

Data is typically represented as:
- Graphs (cells and nets),
- Images/feature maps (layout grids),
- Tabular features (design metrics).

### Benchmark Tasks

| Task                               | Description                                   |
| ---------------------------------- | --------------------------------------------- |
| **Placement prediction**           | Predict good macro or standard-cell placement |
| **Congestion estimation**          | Predict routing congestion before routing     |
| **Timing prediction**              | Estimate critical paths / delay               |
| **Power or wirelength estimation** | Predict QoR metrics                           |
| **Design classification**          | Identify design difficulty or bottlenecks     |


### Baseline Models
The project provides:
- CNN baselines (for layout grids),
- GNN baselines (for netlists),
- Hybrid models (graph + image),
- Training scripts and metrics.

