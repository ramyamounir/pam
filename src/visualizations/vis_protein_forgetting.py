import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd


def calc_bwts(vs):
    past = []
    results = []
    for v in vs:
        if len(v) == 1: continue
        past.extend(v[:-1])
        results.append(sum(past)/len(past))
    return results

def calc_bwt(results):
    counter = 0
    s = 0.0
    for i in results:
        if len(i) == 0: continue
        s+= sum(i[:-1])
        counter += len(i[:-1])

    return s/counter



data = json.load(open('results/protein_forgetting/run_002/results.json', 'r'))
results = {k:calc_bwts(v) for k, v in data.items()}

df = pd.DataFrame(results)
df = df.rename(columns={
    'PAM-1': 'PAM k=1', 
    'PAM-8': 'PAM k=8', 
    'PAM-16': 'PAM k=16', 
    'PAM-24': 'PAM k=24', 
    'PC-1': 'tPC', 
    'HN-1-50': 'AHN d=1 s=$N/2$', 
    'HN-2-50': 'AHN d=2, s=$N/2$'})
df.index = [2, 3, 4, 5, 6, 7, 8, 9, 10]
df.drop(columns=['AHN d=1 s=$N/2$', 'AHN d=2, s=$N/2$'], inplace=True)
print(df)


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
plt.title('Backward Transfer Error Vs. Protein Sequences')
plt.xlabel(r'Number of Protein Sequences')
plt.ylabel(r'BWT Error')

plt.show()  # Show the plot

