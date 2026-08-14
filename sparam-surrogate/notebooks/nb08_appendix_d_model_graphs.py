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
# # Appendix D: selected model structures
#
# This notebook loads the selected artifacts produced by NB03--NB06 and displays
# their model structure. It exports Keras topology diagrams for the portfolio
# and shows scikit-learn estimator diagrams for the non-neural models. It does
# not train, tune, or select a model.

# %%
from __future__ import annotations

import json

import joblib
import keras
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display
from sklearn import set_config
from sklearn.tree import plot_tree

import sparam_surrogate.models.curve_neural  # noqa: F401
import sparam_surrogate.models.full_smatrix  # noqa: F401
from sparam_surrogate.config import PROJECT_ROOT

# The model-module imports register custom layers stored in the saved models.

# %%
selected_path = PROJECT_ROOT / "outputs" / "models" / "selected.json"
if not selected_path.is_file():
    raise FileNotFoundError(f"Selected-model registry not found: {selected_path}")

portfolio_root = PROJECT_ROOT.parent / "project_portfolio"
figure_dir = portfolio_root / "media" / "appendix_d"
evidence_dir = portfolio_root / "evidence"
figure_dir.mkdir(parents=True, exist_ok=True)
evidence_dir.mkdir(parents=True, exist_ok=True)

selected_registry = json.loads(selected_path.read_text(encoding="utf-8"))["models"]

# %% [markdown]
# ## Selected non-neural estimators
#
# Scikit-learn estimators do not have neural-network layers. Their diagrams show
# ordered preprocessing and prediction stages. Each model below is loaded from
# its selected saved artifact. No estimator is fitted again.

# %%
set_config(display="diagram")

# %% [markdown]
# ### Scalar Ridge

# %%
scalar_ridge_selected = selected_registry["scalar_ridge"]
scalar_ridge_path = PROJECT_ROOT / scalar_ridge_selected["artifact_path"]
scalar_ridge_wrapper = joblib.load(scalar_ridge_path)
display(scalar_ridge_wrapper.pipeline)

# %% [markdown]
# The diagram contains two processing stages. It is not a neural network.
#
# 1. `StandardScaler` prepares the 11 inputs. Ten values describe the design.
#    The final value is frequency. `with_mean=True` subtracts each training
#    mean. `with_std=True` divides by each training standard deviation.
#    `copy=True` prevents the scaler from changing the input array in place.
# 2. `Ridge` maps the scaled inputs to one insertion-loss value. The selected
#    penalty is `alpha=0.01`. The penalty limits very large coefficients.
#    `fit_intercept=True` adds a constant offset. `solver="auto"` lets
#    scikit-learn choose the numerical solver. `tol=0.0001` is its convergence
#    tolerance. The coefficients may be positive or negative.
#    `copy_X=True` protects the input matrix. `max_iter=None` leaves the
#    iteration limit to the selected solver. `random_state=None` has no effect
#    on the deterministic solver used for this fitted model.
#
# The model is linear after scaling. It therefore cannot learn local bends or
# interactions unless those patterns are already represented by the inputs.

# %% [markdown]
# ### Vector Ridge

# %%
vector_ridge_selected = selected_registry["vector_ridge"]
vector_ridge_path = PROJECT_ROOT / vector_ridge_selected["artifact_path"]
vector_ridge_wrapper = joblib.load(vector_ridge_path)
display(vector_ridge_wrapper.pipeline)

# %% [markdown]
# This diagram has the same two stages as Scalar Ridge.
#
# 1. `StandardScaler` applies training-fitted centring and scaling to the same
#    11 design-and-frequency inputs.
# 2. `Ridge` uses the selected `alpha=0.01`. It returns six insertion-loss
#    values instead of one. Scikit-learn fits one coefficient vector for each
#    output. The six outputs share a call, but they do not share learned hidden
#    features.
#
# The other Ridge settings retain their defaults. An intercept is fitted. The
# solver is selected automatically. The tolerance is `0.0001`. There is no
# explicit iteration limit. Coefficient signs are unrestricted. The model
# remains linear for every output.

# %% [markdown]
# ### Polynomial Ridge

# %%
polynomial_ridge_selected = selected_registry["polynomial_ridge"]
polynomial_ridge_path = PROJECT_ROOT / polynomial_ridge_selected["artifact_path"]
polynomial_ridge_wrapper = joblib.load(polynomial_ridge_path)
display(polynomial_ridge_wrapper.pipeline)

# %% [markdown]
# This pipeline contains four stages.
#
# 1. `input_scaler` standardises the 11 original inputs using training data.
# 2. `polynomial` expands every input to its first, second and third powers.
#    The selected degree is 3. The width therefore grows from 11 to 33.
#    The transformer does not create products between different inputs.
# 3. `feature_scaler` standardises the 33 expanded values. This stops high
#    powers from dominating only because they have larger numerical ranges.
# 4. `Ridge` maps those values to six outputs. The selected penalty is
#    `alpha=50`. The larger penalty controls the extra polynomial capacity.
#    It fits an intercept. Its solver and stopping settings retain the same
#    defaults as the two Ridge baselines.
#
# The model can represent smooth univariate curvature. It cannot explicitly
# represent terms such as pitch multiplied by trace width.

# %% [markdown]
# ### Random Forest

# %%
random_forest_selected = selected_registry["random_forest"]
random_forest_path = PROJECT_ROOT / random_forest_selected["artifact_path"]
random_forest_wrapper = joblib.load(random_forest_path)
random_forest = random_forest_wrapper.regressor
display(random_forest)

forest_metadata_path = PROJECT_ROOT / random_forest_selected["metadata_path"]
forest_metadata = json.loads(forest_metadata_path.read_text(encoding="utf-8"))
forest_features = forest_metadata["data_interface"]["input_features"]

figure, axis = plt.subplots(figsize=(18, 10))
plot_tree(
    random_forest.estimators_[0],
    max_depth=3,
    feature_names=forest_features,
    filled=False,
    rounded=True,
    impurity=False,
    proportion=True,
    precision=2,
    fontsize=7,
    ax=axis,
)
axis.set_title("Random Forest: first of 128 trees, truncated after depth 3")
figure.tight_layout()
display(figure)
plt.close(figure)

# %% [markdown]
# The estimator diagram describes the complete ensemble. The second plot shows
# only the first tree. It stops after depth 3 so that the upper decisions remain
# readable. It is not a picture of all 128 trees.
#
# - `n_estimators=128` means that the model averages 128 regression trees.
# - `min_samples_leaf=2` prevents a terminal region from containing one training
#   sample. This slightly smooths each tree.
# - `max_depth=None` allows a tree to grow until another stopping rule applies.
# - `criterion="squared_error"` selects splits that reduce squared prediction
#   error. `min_samples_split=2` permits a node with two samples to be split.
# - `bootstrap=True` gives each tree a resampled training set.
# - `max_features=1.0` makes all 11 inputs available when a split is considered.
# - `max_samples=None` gives each bootstrap sample the training-set sample count.
# - `random_state=128` makes the fitted ensemble reproducible.
# - `n_jobs=-1` uses all available processor cores during fitting and inference.
# - No extra cost-complexity or minimum-impurity pruning is requested.
#   Out-of-bag scoring and warm-start training are also disabled.
#
# Each displayed tree node tests one input against a threshold. The branches
# send samples to smaller regions. A leaf stores a six-output mean. Averaging
# the trees reduces the sensitivity to any one bootstrap sample.

# %% [markdown]
# ## Selected neural-model topologies
#
# Keras exposes each layer, tensor shape, activation and trainability flag. The
# diagrams do not show preprocessing owned by the Python wrappers. They also do
# not show inverse scaling after prediction.

# %%
def plot_selected_neural_model(model_name: str) -> dict[str, object]:
    """
    Plot one selected Keras model and return its inventory record.
    """
    selected = selected_registry[model_name]
    artifact_path = PROJECT_ROOT / selected["artifact_path"]
    model = keras.models.load_model(artifact_path, compile=False)
    output_path = figure_dir / f"{model_name}_topology.png"

    keras.utils.plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=220,
        show_layer_activations=True,
        show_trainable=True,
    )
    display(Image(filename=str(output_path), width=900))

    return {
        "model_name": model_name,
        "model_label": selected["model_label"],
        "run_id": selected["run_id"],
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "parameter_count": model.count_params(),
        "topology_figure": output_path.relative_to(portfolio_root).as_posix(),
    }


inventory_rows: list[dict[str, object]] = []

# %% [markdown]
# ### Point-Wise Neural MLP

# %%
neural_mlp_inventory = plot_selected_neural_model("neural_mlp")
inventory_rows.append(neural_mlp_inventory)

# %% [markdown]
# This multilayer perceptron (MLP) processes one design and one frequency at a
# time. Its wrapper standardises the inputs and targets before the Keras graph.
#
# 1. `design_frequency_features` receives 11 values. Ten describe the design.
#    One gives the frequency. The input layer has no trainable parameters.
# 2. `dense_128_a` connects the 11 inputs to 128 units. It has 1,536 trainable
#    weights and biases. The rectified linear unit (ReLU) activation introduces
#    nonlinearity.
# 3. `dense_128_b` maps 128 units to another 128 units. It has 16,512 trainable
#    parameters and uses ReLU.
# 4. `dense_64` reduces the representation to 64 units. It has 8,256 trainable
#    parameters and uses ReLU.
# 5. `s_db_outputs` returns six insertion-loss values. Its linear activation
#    does not clip the output range. It has 390 trainable parameters.
#
# The Keras graph has 26,694 trainable parameters. Inverse target scaling takes
# place after this graph.

# %% [markdown]
# ### Polynomial-Input Neural MLP

# %%
polynomial_mlp_inventory = plot_selected_neural_model("polynomial_neural_mlp")
inventory_rows.append(polynomial_mlp_inventory)

# %% [markdown]
# This network uses the same hidden layers as the point-wise MLP. Its wrapper
# first standardises the 11 original inputs. It then expands each input to
# powers 1 through 5 and standardises the 55 expanded values. The expansion
# does not include products between different features. The six training
# targets are also standardised.
#
# 1. `design_frequency_features` receives the 55 expanded and standardised
#    values. The powers-only transformation is outside the Keras graph.
# 2. `dense_128_a` maps 55 values to 128 ReLU units. The wider input raises this
#    layer to 7,168 trainable parameters.
# 3. `dense_128_b` keeps 128 ReLU units. It has 16,512 parameters.
# 4. `dense_64` reduces the representation to 64 ReLU units. It has 8,256
#    parameters.
# 5. `s_db_outputs` uses 390 parameters to return six linear outputs.
#
# The graph has 32,326 trainable parameters. The increase comes entirely from
# the wider first layer. The wrapper reverses target scaling after prediction.

# %% [markdown]
# ### Whole-Curve Neural Model

# %%
curve_neural_inventory = plot_selected_neural_model("curve_neural")
inventory_rows.append(curve_neural_inventory)

# %% [markdown]
# This model predicts six complete 200-point curves from one design. It uses a
# second input containing fixed frequency coordinates.
#
# 1. `design_parameters` receives the 10 scaled design values.
# 2. `dense_encoder` maps them to 32 ReLU units. It has 352 parameters.
# 3. `dense_projection` creates 800 ReLU values. It has 26,400 parameters.
# 4. `initial_curve_features` reshapes those values to 25 positions with 32
#    channels. Reshaping has no trainable parameters.
# 5. `upsample_1` uses a width-5 transposed convolution. Stride 2 expands the
#    sequence to 50 positions and retains 32 channels. It has 5,152 parameters.
# 6. `upsample_2` expands the sequence to 100 positions and 16 channels. It has
#    2,576 parameters and uses ReLU.
# 7. `upsample_3` expands it to 200 positions and 8 channels. It has 648
#    parameters and uses ReLU.
# 8. `frequency_features` supplies nine fixed values at every frequency. These
#    are normalised frequency and four sine/cosine pairs. They are not trained.
# 9. `add_frequency_features` concatenates the 8 learned channels and 9 fixed
#    channels. The result has 17 channels and no new parameters.
# 10. `curve_refinement` applies a width-5 convolution at all 200 positions. It
#     returns 8 ReLU channels and has 688 parameters.
# 11. `insertion_loss_outputs` applies a width-1 convolution. It returns six
#     linear insertion-loss curves and has 54 parameters.
#
# The Keras graph has 35,870 trainable parameters. Design and target scaling are
# outside the graph. The curve-aware loss is also not a layer in this diagram.

# %% [markdown]
# ### Reciprocal Full-S-Matrix Neural Model

# %%
full_smatrix_inventory = plot_selected_neural_model("full_smatrix_neural")
inventory_rows.append(full_smatrix_inventory)

# %% [markdown]
# This model predicts the complex response at every frequency. The Keras graph
# ends with the unique reciprocal entries. Wrapper code constructs the full
# 12-by-12 matrix afterwards.
#
# Before the graph, the wrapper standardises the 10 design inputs. It scales
# each complex target entry using one training-set root-mean-square magnitude.
# The real and imaginary channels of an entry share that scale.
#
# 1. `design_parameters` receives the 10 scaled design values.
# 2. `repeat_design_over_frequency` copies them across 200 frequencies. It has
#    no trainable parameters and produces a 200-by-10 tensor.
# 3. `fixed_frequency_features` produces 41 fixed values at each frequency.
#    They contain normalised frequency, eight Fourier values and 32 radial-basis
#    values. This layer has no trainable weights.
# 4. `design_and_frequency` concatenates the 10 repeated design values and 41
#    frequency values. It produces 51 features at every frequency.
# 5. `input_projection` maps those 51 features to 128 ReLU features. It has
#    6,656 trainable parameters. Each dense operation is applied separately at
#    all 200 frequency positions.
# 6. Each of the three residual blocks contains four operations. A 128-unit
#    ReLU dense layer learns a transformation. A second 128-unit linear layer
#    prepares a correction. `Add` combines that correction with the block
#    input. A final ReLU activation produces the block output. Each dense layer
#    has 16,512 parameters. The add and activation operations have none.
# 7. `scaled_real_imag_outputs` maps 128 features to 156 linear outputs at each
#    frequency. It has 20,124 parameters. The 156 values represent real and
#    imaginary parts of 78 unique upper-triangle entries.
#
# The graph has 125,852 trainable parameters. After the graph, the wrapper
# reverses complex scaling. It mirrors the upper triangle to guarantee
# reciprocity. Those steps are not visible in the Keras topology.

# %% [markdown]
# ## Selected neural-model inventory

# %%
inventory = pd.DataFrame(inventory_rows)
inventory_path = evidence_dir / "appendix_d_neural_model_inventory.csv"
inventory.to_csv(inventory_path, index=False)
display(inventory)

# %% [markdown]
# Every displayed structure is derived from the selected-model registry and
# saved model files. Only the neural graphs are exported for Appendix D. The
# scikit-learn diagrams and representative tree remain supporting notebook
# views. Re-run this notebook after changing a selected artifact. Do not edit
# the exported images manually.
