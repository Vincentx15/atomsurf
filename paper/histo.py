import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
plt.rc('grid', color='grey', alpha=0.2)

# Data for both methods and metrics
methods = ['GEP', 'AtomSurf']
x_labels = [r'Ab AuROC', r'Ag AuROC', r'Ab MCC', r'Ag MCC']

palette = sns.color_palette()
color_gep = palette[3]
color_as = palette[0]

# Updated values for each method and metric
AuROC_methods = {
    'GEP': [0.80, 0.72],
    'AtomSurf': [0.86, 0.73]
}
MCC_methods = {
    'GEP': [0.28, 0.18],
    'AtomSurf': [0.47, 0.18]
}

# Combine both AuROC and MCC data into a single list
method1_auroc = AuROC_methods['GEP']
method2_auroc = AuROC_methods['AtomSurf']
method1_mcc = MCC_methods['GEP']
method2_mcc = MCC_methods['AtomSurf']

x = np.arange(len(x_labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(figsize=(8, 7))

# Plot AuROC for the first method (GEP)
ax1.bar(x[:2] - width / 2, method1_auroc, width, label=r'GEP', color=color_gep)
# Plot AuROC for the second method (AtomSurf)
ax1.bar(x[:2] + width / 2, method2_auroc, width, label=r'\texttt{AtomSurf}', color=color_as)

# Set the x-axis labels and ticks
ax1.set_ylabel(r'AuROC')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.tick_params(axis='y')
# Set y-axis limits for AuROC to start at 0.5
ax1.set_ylim(0.5, 1.0)

# Create a second y-axis for MCC on the right and plot MCC
ax2 = ax1.twinx()
ax2.bar(x[2:] - width / 2, method1_mcc, width, color=color_gep)
ax2.bar(x[2:] + width / 2, method2_mcc, width, color=color_as)
# Set the second y-axis labels and ticks
ax2.set_ylabel(r'MCC')
ax2.tick_params(axis='y')

# Add a vertical line
midpoint = len(x_labels) / 2 - 0.5  # Position in the middle of the x-axis labels
ax1.axvline(midpoint, ymin=0.02, ymax=0.98, color='gray', linestyle='--', linewidth=2)

# Add legends for both y-axes
fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.9))

# Adjust layout
fig.tight_layout()

plt.savefig('histogram.pdf')

# Show plot
plt.show()
