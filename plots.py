import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

def plot_boxplot_impact_top1(df, variable):
    grouped_data = df.groupby(variable)
    labels = df[variable].unique().tolist()
    groups = [grouped_data.get_group(label)['top1'] for label in labels]
    # Create a figure and axes for the plot
    fig, ax = plt.subplots()
    # Create boxplots for 'M' and 'F' values in gender column
    ax.boxplot(groups, labels=labels)
    # Set the title and labels for the plot
    ax.set_title(f'Top 1 Boxplot by {variable}')
    ax.set_xlabel(variable)
    ax.set_ylabel('Top 1 Value')
    # Save the plot
    plt.savefig(f'plots/1_{variable}.png')
    # Save the plot in SVG format
    plt.savefig(f'plots/1_{variable}.svg')
    # Show the plot
    plt.show()
    return None

# def collapse_columns(df, from_column1, from_column2, to_column):
#     input_df = df.copy()
#     df_move1 = df.copy()
#     df_move2 = df.copy()
#     df_move1[to_column] = df_move1[from_column1]
#     df_move2[to_column] = df_move2[from_column2]
#     del df_move1[from_column1]
#     del df_move1[from_column2]
#     del df_move2[from_column1]
#     del df_move2[from_column2]
#     del input_df[from_column1]
#     del input_df[from_column2]
#     input_df = pd.concat([input_df, df_move1, df_move2], ignore_index=True)
#     return input_df

# Read the csv file and create a dataframe
df = pd.read_csv('data/all_data_preprocessed.csv', sep=';', dtype={'age': 'str', 'class':'str', 'km_driven': 'str'})
df['class'] = pd.Categorical(df['class'], ["1", "4", "9", "18"])

demographic_features = ['gender', 'birthplace', 'age', 'city', 'marital_status', 'education', 'profession']
driver_features = ['car', 'km_driven', 'class']
features = demographic_features + driver_features
column_prices = ['C1/a', 'C1/b', 'C1/c', 'C2/a', 'C2/b', 'C2/c', 'C3/a', 'C3/b', 'C3/c', 'C3/d', 'C4/a', 'C5/a', 'C5/b', 'C6/a']
output_variability_companies_a = ['C1/a', 'C2/a', 'C3/a', 'C4/a', 'C5/a', 'C6/a']
output_variability_companies_any = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
df['C1'] = np.where(df[['C1/a', 'C1/b', 'C1/c']].notnull().any(axis=1), 1, 0)
df['C2'] = np.where(df[['C2/a', 'C2/b', 'C2/c']].notnull().any(axis=1), 1, 0)
df['C3'] = np.where(df[['C3/a', 'C3/b', 'C3/c', 'C3/d']].notnull().any(axis=1), 1, 0)
df['C4'] = np.where(df[['C4/a']].notnull().any(axis=1), 1, 0)
df['C5'] = np.where(df[['C5/a', 'C5/b']].notnull().any(axis=1), 1, 0)
df['C6'] = np.where(df[['C6/a']].notnull().any(axis=1), 1, 0)


# Replace labels for visualization purposes
df = df.replace(
    {'birthplace': 
                {'Milan':'MI', 'Rome':'RO', 'Naples':'NA', 'China':'CN', 'Morocco':'MA'},
    'city':
                {'Milan':'MI', 'Naples':'NA'},  
    'education':
                {'Master':'MSc', 'Without a qualification':'WaQ'},
    'profession':
                {'Employee':'Emp', 'Looking for a job':'LfaJ'},
    'marital_status':
                {'Married':'Mar', 'Single':'Sin', 'Widow':'Wid'}
    })

df['top1'] = df[column_prices].min(axis=1)
df['top2'] = df[column_prices].apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
df['top3'] = df[column_prices].apply(lambda x: x.nsmallest(3).iloc[-1], axis=1)

# df['top123'] = df.apply(lambda row: [row['top1'], row['top2'], row['top3']], axis=1)
df['top123'] = df.apply(lambda row: [value for value in [row['top1'], row['top2'], row['top3']] if not pd.isnull(value)], axis=1)
df.to_csv('data/top123.csv', sep=';', index=False)
# print(df)

exploded_df = df.explode('top123').reset_index(drop=True)
# print(f"exploded_df columns: {exploded_df.columns}")
# print(f"exploded_df columns type: {exploded_df.dtypes}")
exploded_df = exploded_df.infer_objects()
# print(f"exploded_df columns type: {exploded_df.dtypes}")
exploded_df.to_csv('data/exploded_data.csv', sep=';', index=False)
# print(exploded_df['top123'].value_counts())
# print("np.where")
# print(np.where(pd.isnull(exploded_df['top123'])))
# print(df.iloc[3233,:])
# for feature in features:
#     # Create boxplot impact top1
#     plot_boxplot_impact_top1(df, feature)


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
print("top1 boxplot")
# Create a figure with subplots for all the plots figsize=(10, 10)
fig, axs = plt.subplots(1, len(demographic_features),figsize=(25, 10))
# Iterate over each feature and create the corresponding plot
for i, feature in enumerate(demographic_features):
    # Create boxplot impact top1
    grouped_data = df.groupby(feature)
    labels = df[feature].unique().tolist()
    labels.sort()
    groups = [grouped_data.get_group(label)['top1'].values.tolist() for label in labels]
    axs[i].boxplot(groups, labels=labels)
    axs[i].set_title(f'{feature}')
    # axs[i].set_xlabel(feature)
    if i == 0:
        axs[i].set_ylabel('Top 1 Value')
        # print(f"groups is of type {type(groups)}")
        # print(f"groups[0] is of type {type(groups[0])}")
# Adjust the spacing between subplots
# plt.tight_layout()
# Save the plot
plt.savefig('plots/1_top1_all.png')
plt.savefig('plots/1_top1_all.svg')
# Show the plot
# plt.show()

print("top3 boxplot")
# Create a figure with subplots for all the plots figsize=(10, 10)
fig, axs = plt.subplots(1, len(demographic_features), figsize=(25, 10))
# Iterate over each feature and create the corresponding plot
for i, feature in enumerate(demographic_features):
    # Create boxplot impact top1
    grouped_data = exploded_df.groupby(feature)
    labels = exploded_df[feature].unique().tolist()
    groups = [grouped_data.get_group(label)['top123'].values.tolist() for label in labels]    
    axs[i].boxplot(groups, labels=labels)
    axs[i].set_title(f'{feature}')
    # axs[i].set_xlabel(feature)
    if i == 0:
        axs[i].set_ylabel('Top 3 Value')
        # print(f"groups is of type {type(groups)}")
        # print(f"groups[0] is of type {type(groups[0])}")
        # print("grouped_data")
        # print(grouped_data.get_group(labels[0])['top123'].value_counts())
# Adjust the spacing between subplots
# plt.tight_layout()
# Save the plot
plt.savefig('plots/1_top3_all.png')
plt.savefig('plots/1_top3_all.svg')
# Show the plot
# plt.show()

# print("labels")
# print(labels)

# print("groups")
# print(groups)

print("top1&top3 boxplots stacked")
# top1 and top3 boxplots stacked
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
fig, axs = plt.subplots(2, len(features), figsize=(30, 10))
for i, feature in enumerate(features):
    # Create boxplot impact top1
    grouped_data = df.groupby(feature, observed=True)
    labels = df[feature].unique().tolist()
    labels.sort()
    groups_top1 = [grouped_data.get_group(label)['top1'] for label in labels]
    axs[0, i].boxplot(groups_top1, labels=labels)
    axs[0, i].set_xticks([])
    axs[0, i].set_title(f'{feature}', fontsize=24)
    # axs[0, i].set_xlabel(feature)
    if i == 0:
        axs[0, i].set_ylabel('Top 1')
    elif i > 0:
        axs[0, i].set_yticks([])
    # Create boxplot impact top3
    grouped_data = exploded_df.groupby(feature, observed=True)
    labels = exploded_df[feature].unique().tolist()
    labels.sort()
    groups_top3 = [grouped_data.get_group(label)['top123'] for label in labels]
    axs[1, i].boxplot(groups_top3, labels=labels)
    # axs[1, i].set_title(f'{feature}', fontsize=20)
    # axs[1, i].set_xlabel(feature)
    if i == 0:
        axs[1, i].set_ylabel('Top 3')
    elif i > 0:
        axs[1, i].set_yticks([])
# print("groups_top1")
# print(groups_top1)
# print("groups_top3")
# print(groups_top3)
# Adjust the spacing between subplots
plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
# plt.rc('font', size=25)          # controls default text sizes
# plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
# Save the plot
plt.savefig('plots/1_top1-top3_all.png')
plt.savefig('plots/1_top1-top3_all.svg')
plt.savefig('plots/1_top1-top3_all.pdf')
# Show the plot
# plt.show()




#Frequency of quotes observing only /a services
print("frequency of quotes a service")
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
# plt.ylim(0,1)
fig, axs = plt.subplots(len(output_variability_companies_a), len(features), figsize=(30, 12))
for j, company in enumerate(output_variability_companies_a):
    for i, feature in enumerate(features):
        # Group by feature and sum the count of non-null values in output_variability_companies
        grouped_data = df.groupby(feature, observed=True)
        counts = grouped_data[output_variability_companies_a].count().sort_index()
        # transform the counts in frequencies using the values in total_counts
        for col in counts.columns:
            counts[col] = (counts[col] / df.groupby(feature, observed=True).size())*100
        # Plot the counts
        axs[j, i].bar(x=counts.index.to_list(), height=counts[company].values, width=0.7)
        axs[j, i].set_ylim(0, 100)
        if i == 0:
            axs[j, i].set_ylabel(f'f({company})')
            axs[j, i].set_yticks([25,50,75])
            axs[j, i].yaxis.set_major_formatter(mtick.PercentFormatter())
        if i > 0:
            axs[j, i].set_yticks([])
            if i == len(features) - 1:
                axs[j, i].set_yticks([25,50,75])
                axs[j, i].yaxis.set_major_formatter(mtick.PercentFormatter())
                axs[j, i].yaxis.tick_right()
        if j == 0:
            axs[j, i].set_title(f'{feature}', fontsize=28)
            axs[j, i].xaxis.tick_top()
        if j != 0 and j != len(output_variability_companies_a) - 1:
            axs[j, i].set_xticks([])
# Adjust the spacing between subplots
plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plots/3_frequency_a_service.pdf')



#Frequency of quotes for each company
print("frequency of quotes any service")
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
# plt.ylim(0,1)
fig, axs = plt.subplots(len(output_variability_companies_any), len(features), figsize=(30, 12))
for j, company in enumerate(output_variability_companies_any):
    for i, feature in enumerate(features):
        # Group by feature and sum the count of non-null values in output_variability_companies
        grouped_data = df.groupby(feature, observed=True)
        counts = grouped_data[output_variability_companies_any].sum().sort_index()
        # transform the counts in frequencies using the values in total_counts
        for col in counts.columns:
            counts[col] = (counts[col] / df.groupby(feature, observed=True).size())*100
        # Plot the counts
        axs[j, i].bar(x=counts.index.to_list(), height=counts[company].values, width=0.7)
        axs[j, i].set_ylim(0, 100)
        if i == 0:
            axs[j, i].set_ylabel(f'f({company})')
            axs[j, i].set_yticks([25,50,75])
            axs[j, i].yaxis.set_major_formatter(mtick.PercentFormatter())
        if i > 0:
            axs[j, i].set_yticks([])
            if i == len(features) - 1:
                axs[j, i].set_yticks([25,50,75])
                axs[j, i].yaxis.set_major_formatter(mtick.PercentFormatter())
                axs[j, i].yaxis.tick_right()
        if j == 0:
            axs[j, i].set_title(f'{feature}', fontsize=28)
            axs[j, i].xaxis.tick_top()
        if j != 0 and j != len(output_variability_companies_any) - 1:
            axs[j, i].set_xticks([])
# Adjust the spacing between subplots
plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('plots/3_frequency_any_service.pdf')