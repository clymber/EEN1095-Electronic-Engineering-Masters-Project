#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %% [markdown]
# # Exploratory Data Analysis
#
# Here in this notebook, the dataset ["Link on 8 Cavity PCB with two 10×10
# Via-Arrays"](https://www.tet.tuhh.de/en/si-pi-database/) is chosen for exploration.

# %%
"""
Analytical exploration of dataset linkOn8CavityStackBetween10x10Array_19_08_2021
"""

import numpy as np
import numpy.typing  # noqa: F401
import skrf as rf
from IPython.display import Image, display
from matplotlib import pyplot as plt

from sparam_surrogate.config import (
    SurrogateConfig,
    notebook_resource_path,
)
from sparam_surrogate.data import (
    PcbDatasetEDA,
    PcbParameters,
    RawData,
)
from sparam_surrogate.utils.filesystem import directory_tree

cfg = SurrogateConfig.from_config()

# %% [markdown]
# ## 1. Dataset Structure
#
# All datasets from [SI/PI-Database](https://www.tet.tuhh.de/en/si-pi-database/) have
# uniform archive structure, shown by the following procedure:

# %%
print(directory_tree(cfg.dataset.path, max_depth=2, max_children=5))


# %% [markdown]
# - `description.pdf`: Documentation for the dataset, describing the PCB structure,
# simulation setup, parameters, and intended use.
#
# - `parameter.csv`: Table of design or simulation parameters. Each row typically
#   corresponds to one simulation case, often linked by a simulation index such as
#   `SIMU_INDEX`.
#
# - `variation`: Directory containing the simulated S-parameter files for each parameter
#   variation.
#
# - - `simu_0.s12p`, `simu_1.s12p`, etc.: Touchstone files containing S-parameter data
#   for individual simulation cases. The number before .s12p identifies the simulation
#   index.
#
# - - `.s12p`: Touchstone format for a 12-port network, so each file stores
#   frequency-dependent S-parameters for 12 ports.
#
# `class RawData` was developed to embody the fixed structure of the dataset, and to
# provide convenience for further operation:

# %%
rawdata = RawData(cfg.dataset.path, cfg.dataset.nports)
print(f"{len(rawdata.touchstones())} touchstone files found in the dataset.")

# %% [markdown]
# ## 2. Parameter Storage File - `parameter.csv`
#
# The dataset `linkOn8CavityStackBetween10x10Array_19_08_2021` contains geometric
# and material parameters describing PCB interconnect structures and their
# corresponding electromagnetic simulations. The parameters primarily influence
# impedance matching, insertion loss, reflections, coupling, resonance behavior,
# and signal propagation characteristics in the high-speed interconnect.
#
# All parameter variations are stored in the file `parameter.csv`. It is a
# tabular file with multiple columns and rows. The first row has all the
# parameter names as named in the tables and figures of this document. Each row
# is one EM simulation. By the column `SIMU_INDEX` the corresponding network
# parameters are found in the `variation/` folder.
#
# The following procedure shows the table structure, and displays the first and
# last five recoreds along with the PCB parameter variables:

# %%
parameters = PcbParameters(rawdata.parameter_csv)
print(parameters.preview())

# %% [markdown]
# ### 2.1 Feature interpretations
#
# The features naturally divide into several categories:
#
# | Category | Features |
# | -------- | -------- |
# | Material properties | `EPS`, `TAND` |
# | Via geometry | `PITCH`, `VIAR`, `ANTIPADR` |
# | Transmission line geometry | `TRACE_LEN`, `DISTTL`, `TLWIDTH` |
# | Global PCB geometry | `START`, `TDIEL` |

# %% [markdown]
# This is a brief summary of the features:
#
# <table>
# <thead>
# <tr>
# <th>Column</th>
# <th>Unit</th>
# <th>Meaning</th>
# <th>Typical SI Effect</th>
# </tr>
# </thead>
#
# <tbody>
# <tr>
# <td><code>EPS</code></td>
# <td>/</td>
# <td>Relative Permittivity: Dielectric constant of PCB substrate material</td>
# <td>
#     Higher EPS slows wave propagation, reduces wavelength, changes impedance,
#     and shifts resonances</td>
# </tr>
#
# <tr>
# <td><code>TAND</code></td>
# <td>/</td>
# <td>Loss Tangent: Dielectric dissipation factor describing dielectric loss</td>
# <td>
#   Higher TAND increases dielectric attenuation and insertion loss,
#   especially at high frequency
# </td>
# </tr>
#
# <tr>
# <td><code>PITCH</code></td>
# <td>mil</td>
# <td>Via Pitch: Center-to-center spacing between adjacent vias in the array</td>
# <td>
#   Affects electromagnetic coupling, impedance, and crosstalk between vias
# </td>
# </tr>
#
# <tr>
# <td><code>TRACE_LEN</code></td>
# <td>mil</td>
# <td>Trace Length: Length of stripline/interconnect trace between via arrays</td>
# <td>
#     Longer traces generally increase insertion loss, delay, and resonance
#     opportunities
# </td>
# </tr>
#
# <tr>
# <td><code>START</code></td>
# <td>mil</td>
# <td>Start of Via-Array Region: Defines PCB margin surrounding the via arrays</td>
# <td>
#   Influences boundary effects, return current distribution, and overall
#   board dimensions
# </td>
# </tr>
#
# <tr>
# <td><code>VIAR</code></td>
# <td>mil</td>
# <td>Via Radius: Radius of the via barrel</td>
# <td>
#     Changes via inductance/capacitance and therefore impedance and resonance
#     behavior
# </td>
# </tr>
#
# <tr>
# <td><code>ANTIPADR</code></td>
# <td>mil</td>
# <td>
#   Antipad Radius: Radius of the clearance hole around the via in reference planes
# </td>
# <td>Strongly affects parasitic capacitance and impedance discontinuity</td>
# </tr>
#
# <tr>
# <td><code>TDIEL</code></td>
# <td>mil</td>
# <td>Thickness of Dielectric: Thickness of dielectric layer(s)</td>
# <td>
#     Influences characteristic impedance and electromagnetic field distribution
# </td>
# </tr>
#
# <tr>
# <td><code>DISTTL</code></td>
# <td>mil</td>
# <td>
#   Distance between Transmission Lines: Spacing between neighboring traces/links
# </td>
# <td>Larger spacing reduces coupling and crosstalk</td>
# </tr>
#
# <tr>
# <td><code>TLWIDTH</code></td>
# <td>mil</td>
# <td>Transmission Line Width: Width of the stripline/trace</td>
# <td>Strongly affects characteristic impedance and conductor loss</td>
# </tr>
#
# <tr>
# <td><code>SIMU_INDEX</code></td>
# <td>/</td>
# <td>
#   Simulation Index: Unique identifier linking parameter row to corresponding
#   <code>.s12p</code> Touchstone file
# </td>
# <td>Used only for dataset mapping, not a physical parameter</td>
# </tr>
# </tbody>
# </table>

# %% [markdown]
# #### 2.1.1 EPS — Relative Permittivity
#
# This parameter represents the substrate dielectric constant:
# $\varepsilon_r = \frac{\varepsilon}{\varepsilon_0}$, where $\varepsilon$ is
# material permittivity and $\varepsilon_0$ is vacuum permittivity.
#
# Signal propagation velocity approximately follows:
# $v \approx \frac{c}{\sqrt{\varepsilon_r}}$, so higher EPS typically has
# following physical effect:
# - reduces propagation speed,
# - increases delay,
# - shifts resonant frequencies downward, and
# - changes transmission line impedance.
#
# ML (Machine Learning) significance: EPS is often one of the dominant
# parameters affecting phase response, delay, resonance location and insertion
# loss profile.

# %% [markdown]
# #### 2.1.2 TAND — Loss Tangent
#
# Loss tangent describes dielectric energy dissipation:
# $\tan\delta = \frac{\varepsilon''}{\varepsilon'}$, where $\varepsilon''$ is
# imaginary permittivity (loss) and $\varepsilon'$ is real permittivity.
#
# Higher TAND has the following physical effect: (1) increases dielectric
# attenuation, (2) degrades insertion loss, (3) worsens eye diagrams and
# (4) becomes more significant at high frequencies. This parameter is directly
# related to IL target.

# %% [markdown]
# #### 2.1.3 PITCH — Via Pitch
#
# Via pitch is the center-to-center spacing between vias.
#
# Smaller pitch (1) increases electromagnetic coupling, (2) changes parasitic
# capacitance and (3) can increase crosstalk. Larger pitch reduces coupling and
# alters current return paths.
#
# This parameter strongly influences (1) near-end crosstalk (NEXT), (2) far-end
# crosstalk (FEXT) and (3) modal behavior.

# %% [markdown]
# #### 2.1.4 TRACE_LEN — Trace Length
#
# It's the length of interconnect routing. Longer traces (1) increase
# attenuation, (2) increase phase delay, (3) create more resonance
# opportunities, and (4) worsen insertion loss.
#
# Insertion loss typically increases approximately with length:
# $IL \propto \alpha \ell$, where: $\alpha$ is attenuation constant, and $\ell$
# is trace length. This may is one of the strongest predictors for IL.

# %% [markdown]
# #### 2.1.5 START — PCB Edge Clearance / Via-Array Offset
#
# According to the dataset geometry definition:
# $$
# \text{Height} = 2 \cdot START + 9 \cdot PITCH \\
# \text{Width} = 2 \cdot START + TRACE\_LEN + 18 \cdot PITCH
# $$
# START determines the spacing between the via-array structures and the PCB
# edges.
#
# Changing START may influence:
# - PCB cavity dimensions
# - electromagnetic boundary conditions
# - resonance behavior
# - return-current spreading
# - coupling to board edges

# %% [markdown]
# #### 2.1.6 VIAR — Via Radius
#
# Via geometry affects: (1) via inductance, (2) parasitic capacitance and (3)
# impedance discontinuities. Larger vias  may reduce resistance and increase
# capacitance. These effects may impact high-frequency SI (Signal Integrity) behavior.

# %% [markdown]
# #### 2.1.7 ANTIPADR — Antipad Radius
#
# Antipad is the clearance around the via in power/ground planes. Antipad size
# strongly influences parasitic capacitance. Larger antipad reduces capacitance
# and increases impedance.
#
# This parameter often has strong impact on:
# (1) resonance, (2) return loss and (3) impedance discontinuity.

# %% [markdown]
# #### 2.1.8 TDIEL — Dielectric Thickness
#
# Thickness of dielectric layer separating conductors.
#
# Transmission line impedance approximately depends on geometry ratio:
# $Z_0 \sim f\left(\frac{w}{h}, \varepsilon_r\right)$, where $w$ is trace width,
# and $h$ is dielectric height.
#
# TDIEL directly affects: (1) impedance, (2) field confinement and (3) coupling.
#

# %% [markdown]
# #### 2.1.9 DISTTL — Distance Between Transmission Lines
#
# DISTTL measures spacing between neighboring traces. Smaller spacing increases
# coupling and crosstalk, while larger spacing improves isolation.
#
# This parameter is especially important for: (1) differential signaling, and
# (2) multi-lane high-speed channels.

# %% [markdown]
# #### 2.1.10 TLWIDTH — Transmission Line Width
#
# Width of PCB traces.
# - Wider traces result to lower impedance, lower conductor resistance and
#   reduced conductor loss.
#
# - Narrow traces lead to higher impedance, higher current density and more
#   loss.
#
# - This parameter strongly affects characteristic impedance, insertion loss and
# matching quality.

# %% [markdown]
# ### 2.2 Data inspection
#
# The following procedure shows the information about the data frame, from
# perspective of technical implementation:

# %%
parameters.structural_summary()

# %% [markdown]
# The the structural summary reveals several important things about the
# dataset quality, ML readiness and even hidden problems. Details are explored
# in the following subsections:
#
# #### 2.2.1 The dataset is numerically clean
#
# For every features, `7048 non-null` means the dataset has (1) no missing
# values (NaN), (2) no partically corrupted rows, and (3) no incomplete
# parameter records.
#
# **This dataset is clean** for data engineering, so we don't need typical data
# cleaning such as imputation, row filtering or missing-data handling.

# %% [markdown]
# #### 2.2.2 All features are continuous numerical variables
#
# `float64(11)` means every column is stored as floating-point numbers. This
# implies no categorical encoding needed, no string parsing.
#
# The dataset is suitable for solving **continuous regression problem**, and
# aligns well with neural networks, tree regressors and surrogate modeling.

# %% [markdown]
# #### 2.2.3 Data type of `SIMU_INDEX` is wrong
#
# `SIMU_INDEX` originally is of type `float64`, but conceptually it is an
# indentifier, not a physical quantity. This would directly make the wrong
# mapping between index from `parameter.csv` and Touchstone files.
#
# So, **data cleaning is required here to convert to correct data type:
# `int64`**, like:
# ```Python
# parameters["SIMU_INDEX"] = parameters["SIMU_INDEX"].astype(int64)
# ```

# %% [markdown]
# #### 2.2.4 Moderate size of dataset
#
# `7048 entries` suggests that:
#
# | Model Type               | Suitability         |
# | ------------------------ | ------------------- |
# | Linear regression        | Easy                |
# | Random forest            | Good                |
# | Small/medium NN          | Good                |
# | Very deep NN             | Risk of overfitting |
# | Transformer-scale models | Overkill            |
#
# This may implicate the **risk of overfitting** for complex modelling like full
# S-matrix prediction. So the staged approach of "starting simple, and gradually
# increase model complexity" is helpful.

# %% [markdown]
# #### 2.2.5 Memory footprint challenge
#
# Although `memory usage: 605.8 KB` implicates the parameter table is not large,
# and preprocessing cost is low, but the real dataset size is not
# $7048 \times 10$ but $7048 \times N_f \times N_{S}$, where $N_f$ is number of
# frequency points and $N_S$ is number of S-parameters. This is the ML scale of
# the problem.
#
# The heavy computation would be required by Touchstone parsing,
# frequency-domain outputs, and multi-output regression tensors.

# %% [markdown]
# #### 2.2.6 No topology variation
#
# All columns are geometric/material continuous parameters, but no information
# about topology available. The dataset represents one specific topology family
# with parameter variations.
#
# This implicates that the model could not support topology generalization well.

# %% [markdown]
# #### 2.2.7 Data precision
#
# All features use `float64`, and this implies high numerical precision
# retained, and probably sampled from continuous distributions. Further
# investigations are required to identify their distribution.

# %% [markdown]
# #### 2.2.8 Dataset inconsistency issue
#
# The previous result showed that the number of parameter records is 7048, but
# the number of touchstone files is 7639. They were supposed to be the same.
# This suggest that part of the parameter variance records are missing, and
# insidiously some of the Touchstones are also missing.
#
# The following procedure verifys consistency between `parameter.csv` and
# Touchstones:

# %%
rawdata.report_index_consistency()

# %% [markdown]
# The resulting output indicates 609 orphan Touchstones, and 18 missing
# parameter recores.
#
# Real-world dataset are imperfect, and this data integrity checks suggests that
# further data preprocessing is needed.

# %% [markdown]
# #### 2.2.9 Asymmetry between input and response
#
# The dataset only have 10 physical input variables, the output of the models
# potentially have thousands of output values per sample. So the dataset is
# fundamentally low-dimensional input, but high-dimensional frequency response.
#
# So it's a big challenge that whether a compact parameter vector can accurately
# reproduce structured electromagnetic behavious over frequency.

# %% [markdown]
# ### 2.3 Aggregating statistic
#
# The following procedure provides a quick overview of the numerical data in a
# `DataFrame`:

# %%
parameters.statistical_summary()

# %% [markdown]
# Unlike structural summary aforementioned, which focus on how the technical
# implementation builds memory structure and stores data, statistical summary
# describes the data distribution: mean, median, standard deviation, and range.
# In other words, structural summary describes metadata(the data about data), while
# statistical summary describes data itself.

# %% [markdown] vscode={"languageId": "latex"}
# #### 2.3.1 Most parameters are roughly uniformly sampled
#
# For many columns, the median is close to the mean, and the 25% / 75% values
# are fairly symmetric. Examples:
#
# EPS: mean ≈ 4.0009, median ≈ 4.0015  
# PITCH: mean ≈ 60.08, median ≈ 60.12  
# START: mean ≈ 120.16, median ≈ 120.24  
#
# This suggests the design space may have been sampled deliberately, likely to
# cover the parameter range evenly.

# %% [markdown]
# #### 2.3.2 `TRACE_LEN` has a very wide range
#
# TRACE_LEN varies from about 500 mil to 2000 mil. This is likely one of the
# most important parameters for insertion loss, because longer traces usually
# produce more attenuation and delay.

# %% [markdown]
# #### 2.3.3 `DISTTL` may have a skewed or outlier-like distribution
# ```text
# mean = 18.50
# median = 15.75
# max = 56.71
# ```
#
# The max is much larger than the 75% value, 22.00. This suggests a right-skewed
# distribution or some large-spacing cases. This is worth plotting with a
# histogram.

# %% [markdown]
# #### 2.3.4 `TAND` includes zero
# The minimum of `TAND` is 0.000000. Physically, this means some simulations
# assume almost lossless dielectric material. These cases may produce noticeably
# lower insertion loss.
#
# For insertion loss prediction, this is important because:
#
# - higher `TAND` usually increases dielectric loss;
# - `TAND` = 0 cases may produce lower insertion loss;
# - the model should learn the effect of dielectric loss separately from
# geometry effects.
#
# Correlation between TAND and insertion loss at high frequency will be checked later.
# Its effect may be weak at low frequency but stronger at high frequency.

# %% [markdown]
# #### 2.3.5 Feature scaling will be required before neural networks
#
# The feature magnitudes are very different:
#
# * `TAND`: around 0.00–0.02
# * `TRACE_LEN`: around 500–2000
# * `EPS`: around 3.6–4.4
#
# This suggests that `TRACE_LEN` may easily dominate the prediction. A neural
# network would be poorly conditioned without normalization or standardization.

# %% [markdown]
# ### 2.4 Constraint-based validity checking
#
# Some parameters should not be interpreted independently only. For example,
# the antipad radius is always expected larger than the via radius.
#
# #### 2.4.1 Geometry relationship checks
#
# Check whether any parameter records violate:
# - `ANTIPADR > VIAR`
# - `DISTTL > TLWIDTH`

# %%
_ = parameters.check_geometry_relationship()

# %% [markdown]
# #### 2.4.2 Physical range checks

# %%
_ = parameters.check_physical_range()

# %% [markdown]
# #### 2.5.3 Derived feature sanity checks
#
# New features could be derived from existed ones. Further data engineering are
# required to identify which features are helpful, and which ones are
# meaningless.
#
# In the `description.pdf`, the PCB board size is defined as:
#
# $$
# Height = 2 \cdot START + 9 \cdot PITCH \\
# Width = 2 \cdot START + TRACE\_LEN + 18 \cdot PITCH
# $$
#
# This indicates three potential board dimensions, PCB height, width and area,
# could be derived from existed `START`, `PITCH` and `TRACE_LEN` if needed:

# %%
eda = PcbDatasetEDA(parameters)
eda.statistical_summary(["BOARD_HEIGHT", "BOARD_WIDTH", "BOARD_AREA"])

# %% [markdown]
# Some derived ratios may be more physically meaningful than raw parameters
# alone. The other potential derived ratios include:
#
# 1. **Ratio of antipad radius to via radius**
#
# $$
#     \mathrm{ANTIPAD\_TO\_VIA\_RATIO} = \frac{ANTIPADR}{VIAR}
# $$
#
# This ratio measures how much clearance surrounds the via barrel. This ratio
# influences (1) parasitic capacitance, (2) impedance discontinuity, and (3) via
# transition behavior. Larger values generally imply weaker capacitive loading
# and higher impedance around the via.

# %% [markdown]
# 2. **Ratio of tranmission line width to dielectric thickness**
#
# It measures trace width relative to dielectric thickness.
# $$
#     \mathrm{TLWIDTH\_TO\_DIEL\_RATIO} = \frac{TLWIDTH}{TDIEL}
# $$
#
# This ratio is fundamental to transmission-line behavior because characteristic
# impedance approximately depends on:
# $$
#     Z_0 \sim f\left(\frac{w}{h}, \varepsilon_r\right)
# $$
# where: $w$ is trace width, and $h$ is dielectric thickness.
#
# Larger ratios usually correspond to lower characteristic impedance and
# stronger field confinement.

# %% [markdown]
# 3. **Trace aspect ratio**
#
# Measures trace length relative to via-array spacing scale.
# $$
#     \mathrm{TRACE\_ASPECT\_RATIO} = \frac{TRACE\_LEN}{PITCH}
# $$
#
# This gives a rough indication of:
#
# * electrical path elongation,
# * routing scale,
# * resonance opportunity,
# * and accumulated attenuation.
#
# Large values may imply:
#
# * more insertion loss,
# * more distributed transmission-line effects,
# * and stronger frequency-dependent behavior.

# %%
ratio_features = ["ANTIPAD_TO_VIA_RATIO", "TLWIDTH_TO_DIEL_RATIO", "TRACE_ASPECT_RATIO"]
eda.statistical_summary(ratio_features)

# %% [markdown]
# ### 2.5 Visualisation

# %% [markdown]
# #### 2.5.1 Visualize parameter distributions
#
# Histograms are used to inspect the shape of each parameter distribution.
# This helps identify whether the dataset was sampled uniformly, whether
# parameters are skewed, and whether unusual values or clusters exist.

# %%
physical_features = [
    "EPS",
    "TAND",
    "PITCH",
    "TRACE_LEN",
    "START",
    "VIAR",
    "ANTIPADR",
    "TDIEL",
    "DISTTL",
    "TLWIDTH",
]
_ = eda.plot_distribution_histograms(physical_features)

# %% [markdown]
# The histograms show the marginal distributions of the physical input
# parameters in the `parameter.csv` file. Several parameters, including
# `EPS`, `TAND`, `PITCH`, `TRACE_LEN`, `START`, and `TDIEL`, are approximately
# uniformly distributed over bounded intervals. This indicates that the dataset
# was generated using controlled simulation-based parameter variation rather
# than uncontrolled measurement data.
#
# But not all features follow a uniform distribution. The distributions of
# `VIAR` and `ANTIPADR` show opposite tendencies: larger via radii occur less
# frequently, while very small antipad radii are less common. This is physically
# meaningful because the antipad radius must be larger than the via radius, so
# these two parameters should not be interpreted independently. Similarly,
# `DISTTL` is strongly right-skewed, with most samples concentrated at smaller
# transmission-line spacings and relatively few samples at large spacings.
# `TLWIDTH` also shows a non-uniform distribution, suggesting possible geometric
# constraints or design-rule effects.
#
# These observations suggest that the dataset covers a broad parameter space but
# contains dependent geometric relationships between variables. Therefore,
# subsequent exploratory analysis should include both raw parameters and derived
# physical ratios. This is important because machine-learning models trained on
# this dataset will learn not only individual parameter effects, but also the
# constrained design space imposed by PCB geometry.


# %% [markdown]
# #### 2.5.2 Visualize relationships between parameters
#
# The step is to inspect relationships between parameters. This is important
# because several PCB geometry variables are physically constrained and should
# not be interpreted as independent quantities only.
#
# This section uses:
#
# - a correlation heatmap to summarize linear relationships;
# - scatter plots for selected engineering-relevant parameter pairs.

# %%
correlation_features = physical_features + [
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "BOARD_AREA",
    "ANTIPAD_TO_VIA_RATIO",
    "TLWIDTH_TO_DIEL_RATIO",
    "TRACE_ASPECT_RATIO",
]

_ = eda.plot_correlation_heatmap(correlation_features)

# %% [markdown]
# The correlation heatmap provides a compact summary of pairwise linear
# relationships between raw and derived parameters. Strong correlations are
# expected for derived quantities, such as board dimensions and ratio-based
# features, because they are computed directly from the original parameters.
# Therefore, the heatmap should be interpreted as a tool for understanding
# parameter dependency, not as a final feature-selection decision.

# %%
# List the strongest non-diagonal correlations for easier interpretation.
corr_pairs = eda.correlation_pairs(correlation_features)
corr_pairs.head(15)

# %%
physical_constraint_pairs = [
    ("VIAR", "ANTIPADR"),
    ("TLWIDTH", "TDIEL"),
    ("PITCH", "TRACE_LEN"),
    ("DISTTL", "TLWIDTH"),
]

_ = eda.plot_physical_relationships(physical_constraint_pairs)

# %%
_ = eda.plot_board_geometry_verification()

# %% [markdown]
# The scatter plots are separated into two groups.
#
# The first group focuses on physical parameter relationships:
#
# - `ANTIPADR` versus `VIAR` - checks the clearance relationship around vias.
# - `TLWIDTH` versus `TDIEL` - relates trace width to dielectric thickness, which
#   is important for characteristic impedance.
# - `PITCH` versus `TRACE_LEN` - checks whether via-array spacing and trace
#   length were sampled independently.
# - `DISTTL` versus `TLWIDTH` - checks whether line spacing remains larger than
#   trace width.
#
# The second group focuses on derived board-geometry verification:
#
# - `BOARD_HEIGHT` VS `PITCH` - verifies the deterministic board-height definition.
# - `BOARD_WIDTH` VS `TRACE_LEN` - verifies the deterministic board-width definition.
#
# Separating these plots avoids mixing physical feature analysis with derived
# geometry checks. The dashed reference lines mark simple geometric boundary
# conditions. Valid samples are expected to lie on the physically meaningful
# side of these boundaries.

# %%
_ = eda.plot_ratio_relationships()

# %% [markdown]
# The colour-coded scatter plots add the derived ratio as an extra dimension.
# This is useful because the physical effect of two parameters may be better
# represented by their ratio than by either raw parameter alone. For example,
# `ANTIPAD_TO_VIA_RATIO` describes the relative clearance around the via, while
# `TLWIDTH_TO_DIEL_RATIO` is related to transmission-line impedance.

# %% [markdown]
# ## 3. Touchstone files

# %% [markdown]
# Each response file is a 12-port Touchstone network. Touchstone is a plain-text
# RF/microwave file format for storing network parameters over frequency; the
# extension `.s12p` means that each file describes a 12-port network.
#
# In this dataset each file starts with the format header:
#
# ```text
# # GHz S RI R 50.0
# ```
#
# The tokens in this header define how the remaining numeric data should be
# interpreted:
#
# - `#` marks the option line, or format header.
# - `GHz` means the first value in each data block is frequency in gigahertz.
# - `S` says the file stores scattering parameters.
# - `RI` says each complex value is written as real and imaginary parts.
# - `R 50.0` sets the reference impedance to `50 ohm`.
#
# The dataset files then use:
#
# - Frequency grid: 200 values, from `0.5 GHz` to `100.0 GHz` in `0.5 GHz`
#   increments.
# - Response matrix at each frequency: a full complex scattering-parameter
#   matrix, $\mathbf{S}(f) \in \mathbb{C}^{12 \times 12}$.
# - First response targets for modelling: the six through paths declared in
#   configuration, `(7,1)` through `(12,6)`.

# %% [markdown]
# ### 3.1 Touchstone matrix structure
#
# At each frequency, a 12-port Touchstone file stores a scattering matrix:
#
# $$
# \mathbf{S}(f) =
# \begin{bmatrix}
# S_{11}(f) & S_{12}(f) & \cdots & S_{1,12}(f) \\
# S_{21}(f) & S_{22}(f) & \cdots & S_{2,12}(f) \\
# \vdots & \vdots & \ddots & \vdots \\
# S_{12,1}(f) & S_{12,2}(f) & \cdots & S_{12,12}(f)
# \end{bmatrix}
# $$
#
# Each element $S_{ij}(f)$ is the complex ratio between the outgoing wave at
# port $i$ and the incoming wave at port $j$ at frequency $f$. Diagonal terms,
# such as $S_{11}$ and $S_{22}$, are reflection responses, while off-diagonal terms, such as
# $S_{71}$ and $S_{82}$, are transmission or coupling responses between two different ports.
#
# The port configuration is shown in the dataset documentation [2] and is
# collected into `sparam-surrogate/configs/default.json`:

# %%
print("Number of ports: ", cfg.dataset.nports)

# %% [markdown]
# The top view of the PCB helps connect these port numbers to the physical
# structure [2]. Ports 1-6 are on one side of the interconnect, and ports 7-12
# are the corresponding ports on the opposite side.

# %% tags=["remove-input"]
pcb_top_view_img = notebook_resource_path("pcb_top_view.png")
display(Image(filename=str(pcb_top_view_img), width=520))

# %% [markdown]
# The configured response listed above selects the six main through paths from the
# first group of ports to the second group of ports. In each pair `(i, j)`, `i` is
# the receiver/output port and `j` is the source/input port. For example, `(7, 1)`
# selects $S_{7,1}$: the response at port 7 due to excitation at port 1.
#
# Conceptually, the full 12-by-12 matrix can be grouped into reflections,
# insertion-loss paths, and crosstalk paths:
#
# - `R`: reflection, where input and output are the same port.
# - `IL`: insertion-loss or through path between corresponding ports on
#   opposite sides of the structure.
# - `XT`: crosstalk between non-corresponding ports.
#
# $$
# \mathbf{S} =
# \scriptsize
# \begin{bmatrix}
# R & XT & XT & XT & XT & XT & \mathbf{IL} & XT & XT & XT & XT & XT \\
# XT & R & XT & XT & XT & XT & XT & \mathbf{IL} & XT & XT & XT & XT \\
# XT & XT & R & XT & XT & XT & XT & XT & \mathbf{IL} & XT & XT & XT \\
# XT & XT & XT & R & XT & XT & XT & XT & XT & \mathbf{IL} & XT & XT \\
# XT & XT & XT & XT & R & XT & XT & XT & XT & XT & \mathbf{IL} & XT \\
# XT & XT & XT & XT & XT & R & XT & XT & XT & XT & XT & \mathbf{IL} \\
# \mathbf{IL} & XT & XT & XT & XT & XT & R & XT & XT & XT & XT & XT \\
# XT & \mathbf{IL} & XT & XT & XT & XT & XT & R & XT & XT & XT & XT \\
# XT & XT & \mathbf{IL} & XT & XT & XT & XT & XT & R & XT & XT & XT \\
# XT & XT & XT & \mathbf{IL} & XT & XT & XT & XT & XT & R & XT & XT \\
# XT & XT & XT & XT & \mathbf{IL} & XT & XT & XT & XT & XT & R & XT \\
# XT & XT & XT & XT & XT & \mathbf{IL} & XT & XT & XT & XT & XT & R
# \end{bmatrix}
# $$
#
# The matrix above shows both directions of the corresponding through paths.
# The project configuration extracts one direction, `(7,1)` through `(12,6)`,
# which corresponds to the lower-left `IL` entries in this conceptual map.
#
# Because the file uses `RI` format, each complex value is stored as two
# numbers:
#
# $$
# S_{ij}(f) = \operatorname{Re}(S_{ij}) + j\operatorname{Im}(S_{ij})
# $$
#
# The resulting response tensor can be thought of as one $12 \times 12$ matrix for
# every frequency point [1]:

# %% tags=["remove-input"]
smatrix_img = notebook_resource_path("arrays_s_vs_f.png")
display(Image(filename=str(smatrix_img), width=280))

# %% [markdown]
# ### 3.2 Demonstrate `network.s_db`
#
# `skrf.Network.s_db` returns the complex S-parameter network converted to
# magnitude in decibels:
#
# $$
# S_{ij,\mathrm{dB}} = 20 \log_{10} |S_{ij}|
# $$
#
# For this 12-port dataset, `network.s_db` has three dimensions:
#
# 1. frequency point,
# 2. receiver/output port, and
# 3. source/input port.
#
# Therefore, `response_db[:, 6, 0]` means all frequency samples for $S_{7,1}$,
# because the physical Touchstone port pair `(7, 1)` becomes NumPy indices
# `(6, 0)`.
#
# The dB values are usually negative for through paths because they represent
# attenuation: values closer to `0 dB` mean less loss, while more negative
# values mean stronger attenuation.

# %%
example_touchstone = rawdata.touchstones()[0]
network = rf.Network(str(example_touchstone))
# `s_db` is actually a numpy array, the conversion is just for Pylance type consistency.
response_db = np.asarray(network.s_db)

print(f"Loaded example network: {example_touchstone.name}")
print(f"Number of ports: {network.nports}")
print(f"`network.s_db` shape: {response_db.shape}")
print("Shape meaning: (frequency point, receiver port, source port)")
print(f"Frequency span: {network.f[0] / 1e9:g} GHz to {network.f[-1] / 1e9:g} GHz")

# Touchstone port labels are one-based, while NumPy arrays are zero-based.
demo_pair = (7, 1)
demo_curve_db = response_db[:, demo_pair[0] - 1, demo_pair[1] - 1]
print(f"S{demo_pair[0]}{demo_pair[1]} first 5 dB values:", *demo_curve_db[:5])

# %% [markdown]
# ### 3.3 Inspect configured through-path response curves
#
# The magnitude in dB is named directly as `S*_DB`, rather than using an
# insertion-loss label with ambiguous sign. For example, `S7_1_DB` is
# $20 \log_{10} |S_{71}|$. The cell below inspects one example Touchstone file
# only; full training targets are loaded lazily later.

# %%
for pair in cfg.dataset.ports:
    curve = response_db[:, pair[0] - 1, pair[1] - 1]
    print(
        f"S{pair[0]}_{pair[1]}_DB: "
        f"min={curve.min():.3f} dB, "
        f"median={np.median(curve):.3f} dB, "
        f"max={curve.max():.3f} dB"
    )

# %%
frequency_ghz = np.asarray(network.f) / 1e9
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for pair in cfg.dataset.ports:
    curve = response_db[:, pair[0] - 1, pair[1] - 1]
    ax.plot(frequency_ghz, curve, label=f"S{pair[0]}_{pair[1]}_DB")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title(f"Configured through paths for {example_touchstone.name}")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## References
#
# [1] scikit-rf, "`skrf.network.Network.s`," scikit-rf Documentation.
# [Online]. Available:
# https://scikit-rf.readthedocs.io/en/latest/api/generated/skrf.network.Network.s.html.
# [Accessed: May 28, 2026].
#
# [2] M. Schierholz et al., "SI/PI-Database of PCB-Based Interconnects for Machine
# Learning Applications," in IEEE Access, vol. 9, pp. 34423-34432, 2021,
# doi: 10.1109/ACCESS.2021.3061788.
