import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })


# data = {'PAM-8': [0.0, 0.0, 0.0, 0.006526315584778786, 0.03368420898914337, 0.04236841946840286], 'PC-1': [0.0, 0.0, 0.009999999776482582, 0.035947367548942566, 0.047736842185258865, 0.053631577640771866], 'HN-1-5': [0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806], 'HN-1-50': [0.007894736714661121, 0.1742105334997177, 0.2945263087749481, 0.3836315870285034, 0.4445263147354126, 0.5068947076797485], 'HN-2-5': [0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806], 'HN-2-50': [0.0, 0.16673684120178223, 0.285368412733078, 0.3767368495464325, 0.44736841320991516, 0.4981052577495575]}
data = {'PAM-8': [0.0, 0.0, 0.0, 0.0032222222071141005, 0.011111111380159855, 0.03888889029622078], 'PAM-16': [0.0, 0.0, 0.0, 0.0, 0.017777778208255768, 0.03611111268401146], 'PAM-24': [0.0, 0.0, 0.0, 0.0, 0.020555555820465088, 0.0416666679084301], 'PC-1': [0.0, 0.00011111111234640703, 0.004555555526167154, 0.02477777749300003, 0.04188888892531395, 0.049888890236616135]}

# data = json.load(open('results/correlated_sdr_PvK/run_003/results.json', 'r'))
df = pd.DataFrame(data)

df = df.rename(columns={
    'PAM-8': 'PAM k=8', 
    'PAM-16': 'PAM k=16', 
    'PAM-24': 'PAM k=24', 
    'PC-1': 'tPC', 
    'HN-1-5': 'AHN d=1 s=5', 
    'HN-1-50': 'AHN d=1 s=$N/2$', 
    'HN-2-5':'AHN d=2 s=5', 
    'HN-2-50': 'AHN d=2 s=$N/2$'})
# df.index = [0, 1, 2, 3, 4, 5]
df.index = range(0, 120, 20)
# df.drop(columns=['AHN d=1 s=$N/2$', 'AHN d=2 s=$N/2$'], inplace=True)
print(df)

# Plot the DataFrame
sns.set(style="darkgrid")  # Set the style
sns.set_context("poster")
plt.figure(figsize=(8, 6))  # Set the figure size
sns.lineplot(data=df, markers=['o' for _ in range(3)], markersize=20, dashes=False)  # Create the line plot with dot markers

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.xticks(df.index)
plt.title('Error Vs. Noise')
plt.xlabel('% active bits changed')
plt.ylabel('Error')

plt.show()  # Show the plot

