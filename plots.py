import pandas as pd
import matplotlib.pyplot as plt

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

# Read the csv file and create a dataframe
df = pd.read_csv('data/all_data_preprocessed.csv', sep=';')

# Show the frequency of each value in the columns
features = ['gender', 'age', 'birthplace', 'marital_status', 'education', 'profession', 'car', 'km_driven', 'city', 'class']
for feature in features:
    print(df[feature].value_counts())

column_prices = ['C1/a', 'C1/b', 'C1/c', 'C2/a', 'C2/b', 'C2/c', 'C3/a', 'C3/b', 'C3/c', 'C3/d', 'C4/a', 'C5/a', 'C5/b', 'C6/a']

df['top1'] = df[column_prices].min(axis=1)
# df['top2'] = df[column_prices].apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
# df['top3'] = df[column_prices].apply(lambda x: x.nsmallest(3).iloc[-1], axis=1)

for feature in features:
    # Create boxplot impact top1
    plot_boxplot_impact_top1(df, feature)
    

