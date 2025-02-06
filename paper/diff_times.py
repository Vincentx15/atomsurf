import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rc('font', size=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=15)  # fontsize of the tick labels
plt.rc('xtick', labelsize=15)  # fontsize of the tick labels
plt.rc('grid', color='grey', alpha=0.2)

a = torch.load('last.ckpt', map_location='cpu')
values = []
for key, value in a['state_dict'].items():
    if 'diffusion_time' in key:
        values.append(value.detach().cpu().numpy())
import numpy as np

values = np.stack(values)
np.save('diffusion_times.npy', values)

arrays = np.load('diffusion_times.npy')
# Example numpy arrays
array1 = arrays[0]
array2 = arrays[1]
array3 = arrays[2]

# Create a pandas DataFrame
data = pd.DataFrame({
    'Value': np.concatenate([array1, array2, array3]),
    'Block': [r'Block 1'] * len(array1) + [r'Block 2'] * len(array2) + [r'Block 3'] * len(array3)
})

# Plot using seaborn with hue
plt.figure(figsize=(8, 6))
ax = sns.histplot(data=data, x='Value', hue='Block', multiple="dodge", bins=20, kde=True, alpha=0.9)
sns.move_legend(ax, "upper left")
plt.xlabel(r'Diffusion Times')
plt.ylabel(r'Frequency')
plt.grid(True)

plt.savefig('diff_times.pdf')
plt.show()
