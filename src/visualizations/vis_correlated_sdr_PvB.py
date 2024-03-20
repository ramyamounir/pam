
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd



data = json.load(open('results/correlated_sdr_PvB/run_003/results.json', 'r'))
df = pd.DataFrame(data)

df = df.rename(columns={
    'PAM-8': 'PAM k=8', 
    'PC-1': 'tPC', 
    'HN-1-5': 'AHN d=1 s=5', 
    'HN-1-50': 'AHN d=1, s=$N/2$',
    'HN-2-5': 'AHN d=2, s=5',
    'HN-2-50': 'AHN d=2, s=$N/2$',
    })
df.index = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

df.drop(columns=['AHN d=2, s=5', 'AHN d=1 s=5'], inplace=True)


# Plot the DataFrame
sns.set(style="darkgrid")  # Set the style
sns.set_context("poster")
plt.figure(figsize=(8, 6))  # Set the figure size
sns.lineplot(data=df, markers=['o' for _ in range(4)], markersize=20, dashes=False)  # Create the line plot with dot markers

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.yscale('log')
plt.xticks(df.index)
plt.title('Sequence Capacity Vs. Correlation')
# plt.xlabel(r'b=(1-(# unique patterns/N)')
plt.xlabel(r'correlation')
plt.ylabel(r'$P_{max}$')

plt.show()  # Show the plot

