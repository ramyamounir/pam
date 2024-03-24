import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd


data = json.load(open('results/words_gen_iou/run_002/results.json', 'r'))
df = pd.DataFrame(data)
df = df.rename(columns={
    'PAM-1': 'PAM k=1', 
    'PAM-8': 'PAM k=8', 
    'PAM-16': 'PAM k=16', 
    'PAM-24': 'PAM k=24', 
    'PC-1': 'tPC', 
    'HN-1-5': 'AHN d=1 s=5', 
    'HN-2-5': 'AHN d=2 s=5', 
    'HN-1-50': 'AHN d=1 s=$N/2$', 
    'HN-2-50': 'AHN d=2, s=$N/2$',
    })


df.index = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# df.drop(columns=['AHN d=1 s=$N/2$', 'AHN d=2, s=$N/2$'], inplace=True)


# Plot the DataFrame
sns.set(style="darkgrid")  # Set the style
sns.set_context("poster")
plt.figure(figsize=(8, 6))  # Set the figure size
sns.lineplot(data=df, markers=['o' for _ in range(4)], markersize=20, dashes=False)  # Create the line plot with dot markers

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

# plt.yscale('log')
plt.xticks(df.index)
plt.title('IoU of generated words Vs. Number of words')
plt.xlabel(r'Number of words')
plt.ylabel(r'Normalized IoU')

plt.show()  # Show the plot

