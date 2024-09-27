import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def rq1_topn(df, features, column, ylabel):
    """
    Generate boxplots for each feature in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - features (list): A list of column names to use as features.
    - column (str): The column name to use for the boxplot.
    - ylabel (str): The label for the y-axis.

    Returns:
    None
    """
    # Create a figure with subplots for all the plots figsize=(10, 10)
    fig, axs = plt.subplots(1, len(features), figsize=(25, 10))
    # Iterate over each feature and create the corresponding plot
    for i, feature in enumerate(features):
        grouped_data = df.groupby(feature, observed=True)
        labels = df[feature].unique().tolist()
        if feature != 'class':
            labels.sort()
        groups = [grouped_data.get_group(label)[column].values.tolist() for label in labels]
        axs[i].boxplot(groups, labels=labels)
        axs[i].set_title(f'{feature}')
        if i == 0:
            axs[i].set_ylabel(ylabel)
    # Save the plot
    plt.savefig(f'plots/1_{column}_all.pdf')
    return

def rq1_topm_topn(df1, df2, features, column1='top1', column2='top123', ylabel1='Top 1', ylabel2='Top 3'):
    """
    Generate stacked boxplots to visualize the impact of different features on topm and topn values.

    Parameters:
    - df1 (pandas.DataFrame): The first DataFrame containing the data for topm values.
    - df2 (pandas.DataFrame): The second DataFrame containing the data for topn values.
    - features (list): A list of features to be analyzed.
    - column1 (str): The column name for topm values in df1 (default: 'top1').
    - column2 (str): The column name for topn values in df2 (default: 'top123').
    - ylabel1 (str): The label for the y-axis of the topm boxplots (default: 'Top 1').
    - ylabel2 (str): The label for the y-axis of the topn boxplots (default: 'Top 3').

    Returns:
    None
    """

    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    
    fig, axs = plt.subplots(2, len(features), figsize=(30, 10))
    
    for i, feature in enumerate(features):
        # Create boxplot impact topm
        grouped_data = df1.groupby(feature, observed=True)
        labels = df1[feature].unique().tolist()
        if feature != 'class':
            labels.sort()
        groups_top1 = [grouped_data.get_group(label)[column1] for label in labels]
        axs[0, i].boxplot(groups_top1, labels=labels)
        axs[0, i].set_xticks([])
        axs[0, i].set_title(f'{feature}', fontsize=24)
        if i == 0:
            axs[0, i].set_ylabel(ylabel1)
        elif i > 0:
            axs[0, i].set_yticks([])
        
        # Create boxplot impact topn
        grouped_data = df2.groupby(feature, observed=True)
        labels = df2[feature].unique().tolist()
        if feature != 'class':
            labels.sort()
        groups_top3 = [grouped_data.get_group(label)[column2] for label in labels]
        axs[1, i].boxplot(groups_top3, labels=labels)
        if i == 0:
            axs[1, i].set_ylabel(ylabel2)
        elif i > 0:
            axs[1, i].set_yticks([])
    
    # Adjust the spacing between subplots
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0, hspace=0)
    # Save the plot
    plt.savefig(f'plots/1_{column1}-{column2}_all.pdf')
    # plt.show()
    return

def rq3_frequency(df, features, companies, aggregation='count', filename='3_frequency.pdf'):
    """
    Plot the frequency of a feature for different companies.

    Parameters:
    - df: DataFrame - The input DataFrame.
    - features: list - The list of features to plot.
    - companies: list - The list of companies to plot.
    - aggregation: str, optional - The aggregation method to use. Default is 'count'.
    - filename: str, optional - The filename to save the plot. Default is '3_frequency.pdf'.

    Returns:
    None
    """
    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
    # plt.ylim(0,1)
    fig, axs = plt.subplots(len(companies), len(features), figsize=(30, 12))
    for j, company in enumerate(companies):
        for i, feature in enumerate(features):
            # Group by feature and sum the count of non-null values in output_variability_companies
            grouped_data = df.groupby(feature, observed=True)
            if aggregation == 'count':
                counts = grouped_data[companies].count().sort_index()
            elif aggregation == 'sum':
                counts = grouped_data[companies].sum().sort_index()
            else:
                raise ValueError('Invalid aggregation method')
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
            if j != 0 and j != len(companies) - 1:
                axs[j, i].set_xticks([])
    # Adjust the spacing between subplots
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'plots/{filename}')
    return

def rq1_diff_boxplots(df):
    """
    Generate boxplots for each row in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
    None
    """
    plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=24)    # fontsize of the tick labels

    fig, ax = plt.subplots(figsize=(24, 8))

    labels = df.apply(lambda row: f"{row['Attribute']}\n{row['Pairs']}", axis=1)
    data = [row[['.25()', '.50()', '.75()']].values for _, row in df.iterrows()]

    boxprops = dict(linewidth=3)
    medianprops = dict(linewidth=3, color='blue')
    wiskerprops = dict(linewidth=3)
    capprops = dict(linewidth=3)

    # Increment the line width of the frame of the figure
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Increment the line width of the ticks in the x and y axis
    ax.tick_params(axis='both', width=3)

    ax.boxplot(data, labels=labels, boxprops=boxprops, medianprops=medianprops, whiskerprops=wiskerprops, capprops=capprops)
    ax.set_ylabel('Distribution of price differences')
    # ax.set_title('Boxplots for Each Row')
    # plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig('plots/diff_boxplots.pdf') #, transparent=True
    plt.savefig('plots/diff_boxplots.png') #, transparent=True
    plt.show()
    