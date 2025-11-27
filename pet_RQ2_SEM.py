#RQ2: What are the mechanisms by which the quality of the pet ownership affects the mental health of youth?

## RQ2: Data preparation for hierarchical SEM model
# 1. Import required libraries and load the dataset

# We import pandas and numpy for data handling and basic cleaning
import pandas as pd
import numpy as np

csv_path = 'C:/Users/huahua-zz/Desktop/group project/data_clean.csv'
df_raw = pd.read_csv(csv_path)

# Quickly inspect the basic structure to "understand" the dataset
print("\nData shape (rows, columns):", df_raw.shape)
print("\nColumn names:\n", df_raw.columns.tolist())
print("\nData types summary:")
print(df_raw.dtypes)

# Display the first few rows for a quick sanity check
df_raw.head()

# 2.Select variables required for RQ2 hierarchical SEM
rq2_vars = [
    # Pet ownership / structure
    'pet_owner_bin',        # whether currently owns a pet (yes/no or 0/1)
    'num_pets',         # number of pets
    'pet_diversity',    # diversity of pet types

    # Pet–owner bond quality (PRS subscales and total score)
    'prs_affective',
    'prs_family',
    'prs_activity',

    # Self-harm / suicide related indicators
    'sbq_total',        # suicide risk (SBQ-R total score)
    'sh_severity',      # self-harm method severity index
    'sh_diversity',     # self-harm method diversity
    'sh_time_weeks',    # time since last self-harm (weeks)

    # Mood symptom indicators
    'hads_anxiety',
    'hads_depression',
]

# Create a working DataFrame containing only RQ2-related variables
df_rq2 = df_raw[rq2_vars].copy()

print("RQ2 data shape:", df_rq2.shape)
df_rq2.head()

# 3. Basic cleaning for RQ2 variables

# Here we perform light cleaning steps that are generally safe before SEM:
# - Inspect missingness across RQ2 variables.
# - Handle specific missing value rules (e.g., PRS subscales set missing to 0).
# - Optionally, drop rows that are completely missing on key MH indicators.

# Check missing value counts for RQ2 variables
missing_counts = df_rq2.isna().sum().sort_values(ascending=False)
print("Missing values per variable (before cleaning):\n", missing_counts)

# Fill missing values in PRS subscales with 0, as specified
# Rationale: 0 is interpreted as 'no affective/family/activity bond' when data are missing.
prs_cols = ['prs_affective', 'prs_family', 'prs_activity']

for col in prs_cols:
    if col in df_rq2.columns:
        df_rq2[col] = df_rq2[col].fillna(0)
    else:
        print(f"Warning: column '{col}' not found in df_rq2; please check the dataset.")

# Verify that PRS subscales no longer contain missing values
print("\nMissing values in PRS subscales after imputation:")
print(df_rq2[prs_cols].isna().sum())

# Save a cleaned copy for downstream SEM modeling 
df_rq2.to_csv('C:/Users/huahua-zz/Desktop/group project/pet_rq2_cleann.csv', index=False)

# 4. Prepare data matrix for hierarchical SEM (RQ2)

# Here we construct a clean DataFrame containing only the observed variables
# that enter the SEM and drop any remaining rows with missing values on
# these key indicators (listwise deletion for the SEM sample).

sem_vars = [
    # Pet ownership / structure
    'pet_owner_bin', 'num_pets', 'pet_diversity',

    # Pet–owner bond quality (PRS subscales)
    'prs_affective', 'prs_family', 'prs_activity',

    # Suicide / self-harm indicators (for MH_suicide)
    'sbq_total', 'sh_severity', 'sh_diversity', 'sh_time_weeks',

    # Mood symptom indicators (for MH_mood)
    'hads_anxiety', 'hads_depression'
]

# Start from the already cleaned RQ2 dataset
df_sem = df_rq2[sem_vars].copy()

# Drop any rows with missing values on SEM indicators
# (after our previous cleaning, there should be very few or none)
df_sem_before = df_sem.shape[0]
df_sem = df_sem.dropna(subset=sem_vars)
df_sem_after = df_sem.shape[0]

print(f"SEM sample size before listwise deletion: {df_sem_before}")
print(f"SEM sample size after listwise deletion:  {df_sem_after}")

# 5. Specify the hierarchical SEM model for RQ2

# We use semopy's lavaan-style syntax to define a hierarchical mechanism-focused SEM.
# The model includes:
#   (1) Measurement layer
#       - MH_suicide as a latent factor measured by suicide/self-harm indicators.
#       - MH_mood    as a latent factor measured by HADS anxiety/depression.
#       - MH_overall as a second-order factor measured by MH_suicide and MH_mood.
#   (2) Structural layer (mechanistic paths)
#       - Ownership/structure → Bond quality: pet_owner, num_pets, pet_diversity
#         predicting each PRS subscale.
#       - Bond quality → Mental health: PRS subscales predicting MH_mood and MH_suicide.
#       - Cross-dimension path: MH_mood → MH_suicide to test whether mood
#         mediates part of the association between bond quality and suicide risk.

sem_model_desc = """
# Measurement layer
MH_suicide =~ sbq_total + sh_severity + sh_diversity + sh_time_weeks
MH_mood    =~ hads_anxiety + hads_depression

# Second-order factor (overall mental health)
MH_overall =~ MH_suicide + MH_mood

# Structural layer: pet ownership/structure → pet–owner bond quality
prs_affective ~ pet_owner_bin + num_pets + pet_diversity
prs_family    ~ pet_owner_bin + num_pets + pet_diversity
prs_activity  ~ pet_owner_bin + num_pets + pet_diversity

# Structural layer: pet–owner bond quality → mental health dimensions
MH_mood       ~ prs_affective + prs_family + prs_activity
MH_suicide    ~ prs_affective + prs_family + prs_activity + MH_mood
"""

# Print the model string so it is easy to inspect or copy into papers/appendix.
print(sem_model_desc)

# 6. Fit the hierarchical SEM model using semopy
!pip install semopy
try:
    from semopy import Model, Optimizer, calc_stats
except ImportError as e:
    raise ImportError(
        "semopy is required to run the SEM. "
        "Install it with 'pip install semopy' and then re-run this cell."
    ) from e

# Initialize the SEM model object with the specified model description
model = Model(sem_model_desc)

# Fit the model to the prepared SEM dataset (df_sem)
# semopy uses maximum likelihood by default. For small samples, convergence
results = model.fit(df_sem)

print("Model fitting results (optimizer report):")
print(results)

# 6. Fit the hierarchical SEM model using semopy
!pip install semopy
try:
    from semopy import Model, Optimizer, calc_stats
except ImportError as e:
    raise ImportError(
        "semopy is required to run the SEM. "
        "Install it with 'pip install semopy' and then re-run this cell."
    ) from e

# Initialize the SEM model object with the specified model description
model = Model(sem_model_desc)

# Fit the model to the prepared SEM dataset (df_sem)
# semopy uses maximum likelihood by default. For small samples, convergence
results = model.fit(df_sem)

print("Model fitting results (optimizer report):")
print(results)

# Extract parameter estimates (loadings, regressions, variances)
param_estimates = model.inspect()
print("\nFirst few parameter estimates:")
display(param_estimates.head(20))

# Compute and display common global fit indices
stats = calc_stats(model)

# Depending on the semopy version, stats may store fit indices as columns
# (with a simple integer index). To avoid KeyError when selecting by row
# labels, we select them as COLUMNS and then optionally transpose.
fit_names = ['n', 'DoF', 'Chi2', 'p-value', 'CFI', 'TLI', 'RMSEA', 'SRMR']

print("\nFull stats table returned by calc_stats (baseline model):")
print(stats)

# Extract a single row of indices
row_baseline = stats.loc['Value']

# Build a compact Series/DataFrame with commonly reported indices.
fit_indices = pd.DataFrame({
    'DoF': [row_baseline.get('DoF')],
    'Chi2': [row_baseline.get('chi2')],
    'p-value': [row_baseline.get('chi2 p-value')],
    'CFI': [row_baseline.get('CFI')],
    'TLI': [row_baseline.get('TLI')],
    'RMSEA': [row_baseline.get('RMSEA')],
    # SRMR is not provided in this semopy stats output; leave as NaN
    'SRMR': [row_baseline.get('SRMR')],
    # Add sample size from the data directly
    'n': [df_sem.shape[0]]
}).T

print("\nSelected global fit indices for the hierarchical SEM:")
print(fit_indices)

# Save parameter estimates to a CSV file for reporting
param_estimates.to_csv('C:/Users/huahua-zz/Desktop/group project/rq2_sem_parameterss.csv', index=False)

# 7. Summarize baseline SEM results

# At this stage we focus on the core RQ2 hierarchical SEM without
# any demographic covariates. This cell simply provides a clean
# summary of the baseline model fitted in Step 6.

print("Baseline hierarchical SEM (no demographic controls)")
print("\nSelected global fit indices:")
print(fit_indices)

print("\nHead of parameter estimates (loadings & paths):")
display(param_estimates.head(30))

# 8. Visualize the hierarchical SEM (path diagram) 

# We construct a simple conceptual path diagram using networkx + matplotlib

import networkx as nx
import matplotlib.pyplot as plt

# Define latent and observed variables in the model
latent_nodes = ['MH_overall', 'MH_suicide', 'MH_mood']
observed_nodes = [
    # Pet ownership / structure
    'pet_owner', 'num_pets', 'pet_diversity',
    # Pet–owner bond quality (PRS subscales)
    'prs_affective', 'prs_family', 'prs_activity',
    # Suicide / self-harm indicators
    'sbq_total', 'sh_severity', 'sh_diversity', 'sh_time_weeks',
    # Mood symptom indicators
    'hads_anxiety', 'hads_depression'
]

# Create a directed graph
G = nx.DiGraph()
G.add_nodes_from(latent_nodes, layer='latent')
G.add_nodes_from(observed_nodes, layer='observed')

# Add measurement paths (latent → observed)
measurement_edges = [
    ('MH_suicide', 'sbq_total'),
    ('MH_suicide', 'sh_severity'),
    ('MH_suicide', 'sh_diversity'),
    ('MH_suicide', 'sh_time_weeks'),
    ('MH_mood', 'hads_anxiety'),
    ('MH_mood', 'hads_depression')
]

# Second-order factor paths
second_order_edges = [
    ('MH_overall', 'MH_suicide'),
    ('MH_overall', 'MH_mood')
]

# Structural paths: pet ownership/structure → PRS
struct_pet_to_prs = [
    ('pet_owner', 'prs_affective'),
    ('num_pets', 'prs_affective'),
    ('pet_diversity', 'prs_affective'),
    ('pet_owner', 'prs_family'),
    ('num_pets', 'prs_family'),
    ('pet_diversity', 'prs_family'),
    ('pet_owner', 'prs_activity'),
    ('num_pets', 'prs_activity'),
    ('pet_diversity', 'prs_activity')
]

# Structural paths: PRS → MH_mood and MH_suicide (plus MH_mood → MH_suicide)
struct_prs_to_mh = [
    ('prs_affective', 'MH_mood'),
    ('prs_family', 'MH_mood'),
    ('prs_activity', 'MH_mood'),
    ('prs_affective', 'MH_suicide'),
    ('prs_family', 'MH_suicide'),
    ('prs_activity', 'MH_suicide'),
    ('MH_mood', 'MH_suicide')
]

all_edges = measurement_edges + second_order_edges + struct_pet_to_prs + struct_prs_to_mh
G.add_edges_from(all_edges)

# Manually specify node positions to emphasize layered structure
# Bottom: pet ownership / structure
# Mid-left: PRS bond quality
# Mid-right: MH latent factors
# Bottom-right: MH observed indicators
# Top: overall mental health
pos = {
    # Top level (second-order factor) - highest position
    'MH_overall': (0, 4.5),

    # Middle-upper level (first-order latent factors)
    'MH_suicide': (-2.0, 3.5),
    'MH_mood': (2.0, 3.5),

    # Pet ownership / structure (bottom layer)
    'pet_owner': (-4, 0.0),
    'num_pets': (-4.5, -0.6),
    'pet_diversity': (-3.5, -0.6),

    # Lower-middle layer: PRS subscales
    'prs_affective': (-3.0, 1.5),
    'prs_family': (-1.5, 1.5),
    'prs_activity': (0.0, 1.5),

    # Upper-middle layer: Mental health observed indicators
    # Suicide / self-harm indicators
    'sbq_total': (-3.0, 2.5),
    'sh_severity': (-2.0, 2.5),
    'sh_diversity': (-1.0, 2.5),
    'sh_time_weeks': (0.0, 2.5),

    # Mood indicators
    'hads_anxiety': (1.0, 2.5),
    'hads_depression': (2.0, 2.5)
}

# Define groups for styling
pet_nodes = ['pet_owner', 'num_pets', 'pet_diversity']
prs_nodes = ['prs_affective', 'prs_family', 'prs_activity']
sh_nodes = ['sbq_total', 'sh_severity', 'sh_diversity', 'sh_time_weeks']
mood_obs_nodes = ['hads_anxiety', 'hads_depression']

fig, ax = plt.subplots(figsize=(15, 12))

# Draw latent variables (orange circles, larger)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=latent_nodes,
    node_color='#FF9933', node_size=4000,
    edgecolors='black', node_shape='o', ax=ax
)

# Draw pet structure variables (green circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=pet_nodes,
    node_color='#228B22', node_size=3500,
    edgecolors='black', node_shape='o', ax=ax
)

# Draw PRS bond-quality variables (blue circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=prs_nodes,
    node_color='#228B22', node_size=3500,
    edgecolors='black', node_shape='o', ax=ax
)

# Draw self-harm/suicide indicators (purple circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=sh_nodes,
    node_color='#9370DB', node_size=3500,
    edgecolors='black', node_shape='o', ax=ax
)

# Draw mood indicators (pink circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=mood_obs_nodes,
    node_color='#9370DB', node_size=3500,
    edgecolors='black', node_shape='o', ax=ax
)

# Draw labels on top of all nodes
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

# Build a lookup table of estimates & p-values from param_estimates
cols = param_estimates.columns
col_lval = 'lval'
col_rval = 'rval'
col_op   = 'op'
col_est  = 'Estimate' if 'Estimate' in cols else 'est'
col_p    = 'p-value' if 'p-value' in cols else 'pval'

edge_info = {}
for _, row in param_estimates.iterrows():
    op = row[col_op]
    if op == '~':
        source = row[col_rval]
        target = row[col_lval]
    elif op == '=~':
        source = row[col_lval]
        target = row[col_rval]
    else:
        continue

    if (source, target) in edge_info:
        continue
    # Convert p-value to float if possible (some semopy versions store it as string)
    raw_p = row[col_p]
    try:
        p_val = float(raw_p)
    except (TypeError, ValueError):
        p_val = None

    edge_info[(source, target)] = {
        'est': row[col_est],
        'p': p_val
    }

# Draw each edge with style reflecting estimate, sign and significance
for source, target in all_edges:
    if source not in pos or target not in pos:
        continue

    x1, y1 = pos[source]
    x2, y2 = pos[target]

    info = edge_info.get((source, target))
    if info is not None:
        est = info['est']
        pval = info['p']
        lw = 1.0 + 2.0 * min(abs(est), 1.5)
        color = 'tab:red' if est < 0 else 'tab:blue'
        sig = (pval is not None) and (pval < 0.05)
        alpha = 1.0 if sig else 0.4
        linestyle = '-' if sig else '--'
    else:
        # No estimate available in param_estimates: draw a light grey edge
        est = None
        pval = None
        lw = 1.0
        color = 'grey'
        alpha = 0.2
        linestyle = ':'

    # Draw arrow
    ax.annotate(
        '',
        xy=(x2, y2), xycoords='data',
        xytext=(x1, y1), textcoords='data',
        arrowprops=dict(arrowstyle='-|>', lw=lw, color=color,
                        alpha=alpha, linestyle=linestyle),
        zorder=1
    )

    # Add label with estimate and p-value at midpoint
    xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if est is not None:
        label = f"{est:.2f}"
        if pval is not None:
            label += f"\n(p={pval:.3f})"
        ax.text(xm, ym, label, fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

ax.set_title('RQ2 Hierarchical SEM (paths with estimates & significance)')
ax.axis('off')
plt.tight_layout()
plt.show()
