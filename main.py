import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import preprocessing
import plotting

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# def plot_boxplot_impact_top1(df, variable):
#     grouped_data = df.groupby(variable)
#     labels = df[variable].unique().tolist()
#     groups = [grouped_data.get_group(label)['top1'] for label in labels]
#     # Create a figure and axes for the plot
#     fig, ax = plt.subplots()
#     # Create boxplots for 'M' and 'F' values in gender column
#     ax.boxplot(groups, labels=labels)
#     # Set the title and labels for the plot
#     ax.set_title(f'Top 1 Boxplot by {variable}')
#     ax.set_xlabel(variable)
#     ax.set_ylabel('Top 1 Value')
#     # Save the plot
#     plt.savefig(f'plots/1_{variable}.png')
#     # Save the plot in SVG format
#     plt.savefig(f'plots/1_{variable}.svg')
#     # Show the plot
#     plt.show()
#     return None

# Defining various features lists
demographic_features = ['gender', 'birthplace', 'age', 'city', 'marital_status', 'education', 'profession']
driver_features = ['car', 'km_driven', 'class']
features = demographic_features + driver_features
column_prices = ['C1/a', 'C1/b', 'C1/c', 'C2/a', 'C2/b', 'C2/c', 'C3/a', 'C3/b', 'C3/c', 'C3/d', 'C4/a', 'C5/a', 'C5/b', 'C6/a']
output_variability_companies_a = ['C1/a', 'C2/a', 'C3/a', 'C4/a', 'C5/a', 'C6/a']
output_variability_companies_any = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# Define the font size for the plot
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# Read the csv file and create a dataframe
df = pd.read_csv('data/all_data_preprocessed.csv', sep=';', dtype={'age': 'str', 'class':'str', 'km_driven': 'str'})

# Preprocess the dataframe
df = preprocessing.preprocess(df, column_prices)

exploded_top3_df = df.explode('top123').reset_index(drop=True)
exploded_top3_df = exploded_top3_df.infer_objects()
# exploded_top3_df.to_csv('data/exploded_data.csv', sep=';', index=False)
# print(exploded_top3_df['top123'].value_counts())
# print("np.where")
# print(np.where(pd.isnull(exploded_top3_df['top123'])))
# print(df.iloc[3233,:])
# for feature in features:
#     # Create boxplot impact top1
#     plot_boxplot_impact_top1(df, feature)


print("top1 boxplot")
plotting.rq1_topn(df, features, 'top1', 'Top 1 Value')

print("top3 boxplot")
plotting.rq1_topn(exploded_top3_df, features, 'top123', 'Top 3 Value')

print("top1&top3 boxplots stacked")
plotting.rq1_topm_topn(df, exploded_top3_df, features, column1='top1', column2='top123', ylabel1='Top 1', ylabel2='Top 3')


#Frequency of quote
print("frequency of quotes _a service")
plotting.rq3_frequency(df, features, output_variability_companies_a, aggregation='count', filename='3_frequency_a_service.pdf')

print("frequency of quotes _any service")
plotting.rq3_frequency(df, features, output_variability_companies_any, aggregation='sum', filename='3_frequency_any_service.pdf')