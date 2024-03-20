import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd



data = {'PAM-8': [5, 28, 80, 190, 346, 539, 780, 1075, 1399, 1801], 'PC-1': [4, 10, 27, 37, 57, 76, 93, 114, 134, 153], 'HN-1-5': [6, 5, 4, 3, 3, 3, 3, 3, 3, 3], 'HN-1-50': [6, 4, 5, 11, 11, 15, 16, 15, 16, 20]}
# data = json.load(open('results/correlated_sdr_PvN/run_002/results.json', 'r'))
df = pd.DataFrame(data)

df = df.rename(columns={
    'PAM-8': 'PAM k=8', 
    'PC-1': 'tPC', 
    'HN-1-5': 'AHN d=1 s=5', 
    'HN-1-50': 'AHN d=1, s=$N/2$'})
df.index = range(10, 110, 10)
print(df)


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
plt.title('Sequence Capacity Vs. Input Size')
plt.xlabel(r'$N$')
plt.ylabel(r'$P_{max}$')

plt.show()  # Show the plot

