import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd


def get_bar_df(data_1, data_10, data_100):

    df_dict = {'Num_words':[], 'Model':[], 'Dataset recall':[]}
    categories = [1, 10, 100]

    for data_ix, data in enumerate([data_1, data_10, data_100]):
        for k, v in data.items():
            df_dict['Num_words'].append(categories[data_ix])
            df_dict['Model'].append(k)
            df_dict['Dataset recall'].append(v[-1])

    return pd.DataFrame(df_dict)


def plot_bar_chart():
    bar_df = get_bar_df(data_1, data_10, data_100)

    bar_df = bar_df[ bar_df['Model'] != 'HN-1-5'  ]
    bar_df = bar_df[ bar_df['Model'] != 'HN-2-5'  ]

    renames = {
    # 'PAM-1': 'PAM k=1', 
    # 'PAM-8': 'PAM k=8', 
    # 'PAM-16': 'PAM k=16', 
    # 'PAM-24': 'PAM k=24', 
    'PC-1': 'tPC', 
    # 'HN-1-5': 'AHN d=1 s=5', 
    # 'HN-2-5': 'AHN d=2 s=5', 
    # 'HN-1-50': 'AHN d=1 s=$N/2$', 
    # 'HN-2-50': 'AHN d=2, s=$N/2$'
    }

    for k, v in renames.items():
        bar_df['Model'] = bar_df['Model'].replace(k, v)


    sns.set(style="darkgrid")  # Set the style
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))  # Set the figure size

    ax = sns.barplot(data=bar_df, x='Model', y='Dataset recall', hue='Num_words', palette='Set2')
    # plt.title('Dataset Recall after 5 Generations', pad=80)
    plt.xlabel('')
    plt.ylabel('Dataset Recall')
    plt.legend(title='Number of Words', bbox_to_anchor=(0.5, 1.16), loc='upper center', ncol=len(bar_df['Model'].unique()))
    # plt.legend(title='Number of Words')
    plt.subplots_adjust(top=0.80)
    # plt.legend(title='Number of Words')

    # # Annotate each bar with its value
    # for p in ax.patches:
    #     ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', xytext=(0, 3), textcoords='offset points', fontsize=9)


    plt.show()



def get_line_df(data_100):

    collect_models = ['PAM-1', 'PAM-8', 'PAM-16', 'PAM-24']
    df_dict = {}

    for k, v in data_100.items():
        if k not in collect_models: continue
        df_dict[k] = v

    line_df = pd.DataFrame(df_dict)
    line_df.index = [1, 2, 3, 4, 5]

    line_df = line_df.rename(columns={
        'PAM-1': 'PAM k=1', 
        'PAM-8': 'PAM k=8', 
        'PAM-16': 'PAM k=16', 
        'PAM-24': 'PAM k=24', 
        })

    return line_df


def plot_line_chart():
    line_df = get_line_df(data_100)


    # Plot the DataFrame
    sns.set(style="darkgrid")  # Set the style
    sns.set_context("poster")
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.lineplot(data=line_df, markers=['o' for _ in range(4)], markersize=20, dashes=False)  # Create the line plot with dot markers

    plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
    plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
    plt.grid(which='both', linestyle='--')

    # plt.yscale('log')
    plt.xticks(line_df.index)
    plt.title('Dataset recall (100 words) Vs. Number of Generations')
    plt.xlabel(r'Number of Generations')
    plt.ylabel(r'Dataset Recall')

    plt.show()  # Show the plot




data_1 = json.load(open('results/words_gen_recall/run_001/results_1.json', 'r'))
data_10 = json.load(open('results/words_gen_recall/run_001/results_10.json', 'r'))
data_100 = json.load(open('results/words_gen_recall/run_001/results_100.json', 'r'))

# plot_bar_chart()
plot_line_chart()
quit()




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

