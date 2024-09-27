import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import preprocessing
import plotting
import discrimination_analysis
import time

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

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
column_prices = ['C1/a', 'C1/b', 'C1/c', 'C1/d', 'C2/a', 'C2/b', 'C2/c', 'C3/a', 'C3/b', 'C3/c', 'C3/d', 'C4/a', 'C5/a', 'C5/b', 'C6/a']
output_variability_companies_a = ['C1/a', 'C2/a', 'C3/a', 'C4/a', 'C5/a', 'C6/a']
output_variability_companies_any = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
output_variability_companies_any_meaningful = ['C2', 'C3', 'C4', 'C6']

# Define the font size for the plot
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

# Read the csv file and create a dataframe
df = pd.read_csv('data/all_data_preprocessed.csv', sep=';', dtype={'age': 'str', 'class':'str', 'km_driven': 'str'})
cp_df = pd.read_csv('data/preprocessed_control_queries.csv', sep=';', dtype={'age': 'str', 'class':'str', 'km_driven': 'str'})
# Preprocess the dataframe

print("Starting preprocessing...")
start_time = time.time()

df = preprocessing.preprocess(df, column_prices, features)
cp_df = preprocessing.preprocess(cp_df, column_prices, features)

end_time = time.time()
execution_time = end_time - start_time
print(f"Preprocessing done in {execution_time:.0f} seconds.")

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

# print("top3 boxplot")
# plotting.rq1_topn(exploded_top3_df, features, 'top123', 'Top 3 Value')

print("top5 boxplot")
plotting.rq1_topn(df, features, 'top5avg', 'Top 5 Value')

# print("top1&top3 boxplots stacked")
# plotting.rq1_topm_topn(df, exploded_top3_df, features, column1='top1', column2='top123', ylabel1='Top 1', ylabel2='Top 3')

print("top1&top5 boxplots stacked")
plotting.rq1_topm_topn(df, df, features, column1='top1', column2='top5avg', ylabel1='Top 1', ylabel2='Top 5')

print("rq2 discrimination analysis - top1")
gd_fem = discrimination_analysis.differences_distribution(df, 'gender', 'F', 'M', 'top1')
bp_ro = discrimination_analysis.differences_distribution(df, 'birthplace', 'RO', 'MI', 'top1')
bp_na = discrimination_analysis.differences_distribution(df, 'birthplace', 'NA', 'MI', 'top1')
bp_ma = discrimination_analysis.differences_distribution(df, 'birthplace', 'MA', 'MI', 'top1')
bp_cn = discrimination_analysis.differences_distribution(df, 'birthplace', 'CN', 'MI', 'top1')
age = discrimination_analysis.differences_distribution(df, 'age', '25', '32', 'top1')
city = discrimination_analysis.differences_distribution(df, 'city', 'NA', 'MI', 'top1')
# pr_emp = discrimination_analysis.differences_distribution(df, 'profession', 'Emp', 'LfaJ', 'top1')
# ed_msc = discrimination_analysis.differences_distribution(df, 'education', 'MSc', 'WaQ', 'top1')
# ms_sin = discrimination_analysis.differences_distribution(df, 'marital_status', 'Sin', 'Wid', 'top1')
ms_sin = discrimination_analysis.differences_distribution(df, 'marital_status', 'Sin', 'Mar', 'top1')
ms_wid = discrimination_analysis.differences_distribution(df, 'marital_status', 'Wid', 'Mar', 'top1')
ed_msc = discrimination_analysis.differences_distribution(df, 'education', 'WaQ', 'MSc', 'top1')
pr_emp = discrimination_analysis.differences_distribution(df, 'profession', 'LfaJ', 'Emp', 'top1')

# control pairs
cp = discrimination_analysis.control_pairs(df, cp_df, features, 'top1')
# Combine the dataframes
rq2_top1_df = pd.concat([gd_fem, bp_ro, bp_na, bp_ma, bp_cn, age, city, ms_sin, ms_wid, ed_msc, pr_emp, cp], ignore_index=True)
rq2_top1_df.to_latex("tables/rq2_discrimination_analysis_top1.tex", index=False, caption='Discrimination Analysis Results', label='table:discrimination_analysis')
print(rq2_top1_df)

print("rq2 discrimination analysis - top5")
gd_fem = discrimination_analysis.differences_distribution(df, 'gender', 'F', 'M', 'top5avg')
bp_ro = discrimination_analysis.differences_distribution(df, 'birthplace', 'RO', 'MI', 'top5avg')
bp_na = discrimination_analysis.differences_distribution(df, 'birthplace', 'NA', 'MI', 'top5avg')
bp_ma = discrimination_analysis.differences_distribution(df, 'birthplace', 'MA', 'MI', 'top5avg')
bp_cn = discrimination_analysis.differences_distribution(df, 'birthplace', 'CN', 'MI', 'top5avg')
age = discrimination_analysis.differences_distribution(df, 'age', '25', '32', 'top5avg')
city = discrimination_analysis.differences_distribution(df, 'city', 'NA', 'MI', 'top5avg')
# pr_emp = discrimination_analysis.differences_distribution(df, 'profession', 'Emp', 'LfaJ', 'top5avg')
# ed_msc = discrimination_analysis.differences_distribution(df, 'education', 'MSc', 'WaQ', 'top5avg')
# ms_sin = discrimination_analysis.differences_distribution(df, 'marital_status', 'Sin', 'Wid', 'top5avg')
ms_sin = discrimination_analysis.differences_distribution(df, 'marital_status', 'Sin', 'Mar', 'top5avg')
ms_wid = discrimination_analysis.differences_distribution(df, 'marital_status', 'Wid', 'Mar', 'top5avg')
ed_msc = discrimination_analysis.differences_distribution(df, 'education', 'WaQ', 'MSc', 'top5avg')
pr_emp = discrimination_analysis.differences_distribution(df, 'profession', 'LfaJ', 'Emp', 'top5avg')
# control pairs
cp = discrimination_analysis.control_pairs(df, cp_df, features, 'top5avg')
# Combine the dataframes
rq2_top5_df = pd.concat([gd_fem, bp_ro, bp_na, bp_ma, bp_cn, age, city, ms_sin, ms_wid, ed_msc, pr_emp, cp], ignore_index=True)
rq2_top5_df.to_latex("tables/rq2_discrimination_analysis_top5.tex", index=False, caption='Discrimination Analysis Results', label='table:discrimination_analysis')
print(rq2_top5_df)
# Merge rq2_top1_df and rq2_top5_df
merged_df = pd.merge(rq2_top1_df, rq2_top5_df, on=['Attribute', 'Pairs'], suffixes=('_top1', '_top5'))
merged_df. replace({'Attribute': {'birthplace': 'Birthplace', 'gender': 'Gender', 'profession': 'Profession', 'education': 'Education', 'marital_status': 'Mar. Stat.', 'age': 'Age', 'city': 'City', 'control pairs 1': '\\multicolumn{2}{l|}{Control pairs (noise) 1}', 'control pairs 2': '\\multicolumn{2}{l|}{Control pairs (noise) 2}', 'control pairs':'\\multicolumn{2}{l|}{Control pairs (noise)}'}}, inplace=True)
merged_df.to_latex("tables/rq2_discrimination_analysis_merged.tex", index=False, caption='Discrimination Analysis Results', label='table:discrimination_analysis')

# Plot the discrimination analysis results
gd_fem_q_n = discrimination_analysis.differences_distribution(df, 'gender', 'F', 'M', 'top5avg', quartiles=True, numeric=True)
bp_na_q_n = discrimination_analysis.differences_distribution(df, 'birthplace', 'NA', 'MI', 'top5avg', quartiles=True, numeric=True)
bp_ma_q_n = discrimination_analysis.differences_distribution(df, 'birthplace', 'MA', 'MI', 'top5avg', quartiles=True, numeric=True)
bp_cn_q_n = discrimination_analysis.differences_distribution(df, 'birthplace', 'CN', 'MI', 'top5avg', quartiles=True, numeric=True)
age_q_n = discrimination_analysis.differences_distribution(df, 'age', '25', '32', 'top5avg', quartiles=True, numeric=True)
city_q_n = discrimination_analysis.differences_distribution(df, 'city', 'NA', 'MI', 'top5avg', quartiles=True, numeric=True)
ms_wid_q_n = discrimination_analysis.differences_distribution(df, 'marital_status', 'Wid', 'Mar', 'top5avg', quartiles=True, numeric=True)
ed_msc_q_n = discrimination_analysis.differences_distribution(df, 'education', 'WaQ', 'MSc', 'top5avg', quartiles=True, numeric=True)
pr_emp_q_n = discrimination_analysis.differences_distribution(df, 'profession', 'LfaJ', 'Emp', 'top5avg', quartiles=True, numeric=True)
rq1_top5_plot_df = pd.concat([gd_fem_q_n, bp_na_q_n, bp_ma_q_n, bp_cn_q_n, age_q_n, city_q_n, ms_wid_q_n, ed_msc_q_n, pr_emp_q_n], ignore_index=True)
print(rq1_top5_plot_df)
plotting.rq1_diff_boxplots(rq1_top5_plot_df)

# RQ3 Frequency of quote
print("frequency of quotes _a service")
plotting.rq3_frequency(df, features, output_variability_companies_a, aggregation='count', filename='3_frequency_a_service.pdf')

print("frequency of quotes _any service")
plotting.rq3_frequency(df, features, output_variability_companies_any_meaningful, aggregation='sum', filename='3_frequency_any_service.pdf')

