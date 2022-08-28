#%% Imports
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from scipy import stats

from lca import LCA

from network_visualization import Grapher

#%% Read data
col_origin = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation_disorder", "peripheral_vascular_disorder", "hypertension_uncomplicated", "hypertension_complicated", "paralysis", "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity", "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression"]
columns = ["CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder", "Peripheral vascular disorder", "Uncomplicated hypertension", "Complicated hypertension", "Paralysis", "Other neurological disorder", "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism", "Renal failure", "Liver disease", "Peptic ulcer disease", "AID/HIV", "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)", "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss", "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia", "Alcohol abuse", "Drug abuse", "Psychoses", "Depression"]
# Pandas doesn't seem to support column names with underscores?
df1: DataFrame = pd.read_excel("data/tbi_admit_icd_age_elix.xlsx") # Spreadsheet with only elix groupings 
df1 = df1[col_origin] # Only take cols of interest
df1 = df1.replace({True: 1, False: 0}) # Convert True to 1 and False to 0
print(df1.shape)
# df1 = df1[df1.sum(axis=1)!=0] # Remove empty rows?
data = df1.to_numpy() # Need conversion before you can use in LCA model

#%% Generate data (optional)

columns = ["C1","C2","C3","C4"]
true_theta = [
    [0.1,0.4,0.9,0.2],
    [0.5,0.9,0.1,0.1],
    [0.9,0.9,0.5,0.9]
]
true_weights = [0.1, 0.5, 0.4]
N = 10000

data = []
for tw,tt in zip(true_weights,true_theta):
    data.append(stats.bernoulli.rvs(p=tt, size=(int(tw*N),len(tt))).tolist())
    
data = np.concatenate(data)

#%% LCA Algorithm

classes = 9
lca = LCA(n_components=classes, tol=1e-10, max_iter=1000)
lca.fit(data)
print(lca.weight)


fig, ax = plt.subplots(figsize=(15,5)) # Plot learning curve
ax.plot(lca.ll_[1:], linewidth=3)
ax.set_title("Log-Likelihod")
ax.set_xlabel("iteration")
ax.set_ylabel(r"p(x|$\theta$)")
ax.grid(True)

#%% Results pt1: Bar plots of elix distributions over clusters
fig, axs = plt.subplots(nrows=lca.theta.shape[0], figsize=(15,lca.theta.shape[0]*10))
axs = axs.ravel()
for i,ax in enumerate(axs):
    ax.bar(range(len(columns)),lca.theta[i,:]) # Single colon in index means fetch everything
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation="vertical")
    ax.set(xlabel = "Elixhauser comorbidity group", ylabel = "Prevalence of elixidity group in latent class",
        title = f"Latent class {i+1}")
    



#%% Results pt2: Stacked horizontal bar plots of elix distributions
lca.theta.shape[0] # Gives number of classes
lca.theta.shape[1] # Gives number of input variables 
probs_classes = [lca.theta[:,i] for i in range(lca.theta.shape[1])] # Returns a list of arrays corresponding to the probabilities of each variable in each class
dict_probs = dict(zip(columns, probs_classes))
print(len(dict_probs))

# Discrete distribution horizontal bar chart https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py
category_names = [f"Endotype {i}" for i in range(1, len(columns)+1)]
results = dict_probs # Dictionary containing comorbidities as keys and its probability distribution across subgroups (categories)


labels = list(results.keys())
data = np.array(list(results.values())) # XXXXXXXXXX This modifies the data variable that's needed later on, should refactor
data_cum = data.cumsum(axis=1)
category_colors = plt.colormaps['RdYlGn'](
    np.linspace(0.15, 0.85, data.shape[1]))
category_colors = plt.colormaps['jet'](
    np.linspace(0.15, 0.85, data.shape[1]))

fig, ax = plt.subplots(figsize=(data.shape[1]*1.75, data.shape[0]*0.5))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data, axis=1).max()) # This sets the xlim to the max value
ax.set(
    xlabel = "Prevalence of comorbidity group",
    ylabel = "Elixhauser comorbidity group",
    title = "?"
    )

for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color)

    r, g, b, _ = color
    text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
    # ax.bar_label(rects, label_type='center', color=text_color) # Bar labels may be messy if bars are small
ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
            loc='upper left', fontsize='small')

plt.show()

#%% Results pt3 (endotype visualization)

df_list = [] # Container for DFs containing elix distributions for each subgroup 
for i in range(lca.theta.shape[0]):
    entry = pd.DataFrame({
        "Name": columns,
        "Value": lca.theta[i,:]
    }) # Need comprehension to package every value in its own list for DF creation
    df_list.append(entry)
print(df_list)

# Radial bar plots
max_height = 100
lower_radius = 50
max_value = 1
n_cols = 3
n_rows = (classes + n_cols - 1) // n_cols
category_colors = plt.colormaps['jet'](
    np.linspace(0.15, 0.85, len(df_list)))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,21), dpi = 300, subplot_kw={'projection': 'polar'})
axs = axs.ravel() # Returns flattened version of array (i.e., array of axes)

for index, ax in enumerate(axs):
    ax: Axes
    df = df_list[index]
    ax.axis('off') # Remove grid
    # max_value = df['Value'].max() # Use for scaling

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    heights = max_height * df.Value/max_value # Returns list of heights

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi/len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index)+1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lower_radius,
        linewidth=2, 
        edgecolor="#FFFFFFFF",
        color=category_colors[index],
        alpha = 0.8)

    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, columns):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=bar.get_height() + lower_radius,
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
    # ax.set_title(f"Endotype {str(index+1)}", y = 1.4, pad = 15) # Use title
    ax.text(0.5, 0.5, f"Endotype {str(index+1)}", horizontalalignment="center", transform=ax.transAxes) # Use text placement to place text at center of plots

#%% Generate annotations
# Generate annotations
data = df1.to_numpy()
df2 = pd.read_excel("data/merged_output_elix_formatted.xlsx") # Merged master spreadsheet with basic clinical info and comorbidity mappings
res = lca.predict(data)
print(res.shape, type(res), res)
entry = pd.DataFrame({"LCA class": res})
print(entry)
# Merge annotations with original df
print(df2.shape)
df2 = pd.concat([df2, entry], axis= 1)
print(df2.shape)
df2.to_excel("data/merged_output_elix_formatted_annotated.xlsx")

#%% Calculating mortality in each cluster
df3 = pd.read_excel("data/merged_output_elix_formatted_annotated.xlsx")
print(df3["DOD"].count())
num_clusters = 8
df_export = pd.DataFrame()
for i in range(num_clusters): 
    df_sub = df3.loc[df3["LCA class"] == i] # Get all rows of this cluster
    num_rows = df_sub.shape[0]
    print(num_rows)
    dead = df_sub["DOD"].count()
    alive = num_rows - dead
    entry = pd.DataFrame({i: [alive, dead]})
    df_export = pd.concat([df_export, entry], axis = 1)
print(df_export)
df_export.to_excel("data/merged_cluster_mortality.xlsx")

#%% Model selection
ks = list(range(2,16))
bics = []
for k in ks:
    print(k)
    lca = LCA(n_components=k, tol=10e-10, max_iter=5000)
    lca.fit(data)
    bics.append(lca.bic)
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(ks, bics, linewidth=3)
ax.grid(True)
ax.set(xlabel = "Number of latent classes", ylabel = "Bayesian Information Criterion (BIC)",
    title = f"Elbow plot")
#%% Elbow plot
y = bics
x = ks

from kneed import KneeLocator
kn = KneeLocator(x, y, curve='convex', direction='decreasing')
print(kn.knee)

plt.xlabel('Number of latent classes')
plt.ylabel('Bayesian Information Criterion (BIC)')
plt.plot(x, y, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
#%% Elbow plot v2
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(x, y, linewidth=3)
ax.grid(True)
ax.set(xlabel = "Number of latent classes", ylabel = "Bayesian Information Criterion (BIC)",
    title = f"Elbow plot")