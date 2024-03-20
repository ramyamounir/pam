import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd

# data = {'PAM-8': [0.0, 0.0, 0.0027894736267626286, 0.0046315789222717285, 0.04210526496171951, 0.046315789222717285], 'PC-1': [0.0, 0.00010526315600145608, 0.007263157982379198, 0.03636842221021652, 0.04899999871850014, 0.052105262875556946], 'HN-1-5': [0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806], 'HN-1-50': [0.015263157896697521, 0.1778947412967682, 0.28736841678619385, 0.3845263123512268, 0.44652631878852844, 0.5002631545066833], 'HN-2-5': [0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806], 'HN-2-50': [0.0, 0.1669473648071289, 0.28473684191703796, 0.37315788865089417, 0.4471578896045685, 0.50031578540802]}

data = {'PAM-8': [5, 28, 80, 190], 'PAM-16': [7, 83, 291, 621], 'PAM-24': [15, 190, 583, 1208], 'HN-2-5': [9, 9, 5, 4], 'HN-2-50': [9, 27, 53, 108]}
# data = json.load(open('results/correlated_sdr_PvK/run_003/results.json', 'r'))
df = pd.DataFrame(data)

df = df.rename(columns={'PAM-8': 'PAM k=8', 'PAM-16': 'PAM k=16', 'PAM-24': 'PAM k=24', 'HN-2-5':'AHN d=2 s=5', 'HN-2-50': 'AHN d=2 s=$N/2$'})
df.index = [10, 20, 30, 40]


# Plot the DataFrame
sns.set(style="darkgrid")  # Set the style
sns.set_context("poster")
plt.figure(figsize=(8, 6))  # Set the figure size
sns.lineplot(data=df, markers=['o' for _ in range(3)], markersize=20, dashes=False)  # Create the line plot with dot markers

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.xticks(df.index)
plt.title('Sequence Capacity Vs. Input Size')
plt.xlabel(r'$N$')
plt.ylabel(r'$P_{max}$')

plt.show()  # Show the plot

