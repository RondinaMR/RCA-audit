import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

def compute_distribution(df, column, attribute_description=None, pairs_description=None, debug=False):
    """
    Compute the distribution of a given column in a DataFrame.
    From statsmodels.stats.descriptivestats.sign_test:
        The signs test returns M = (N(+) - N(-))/2
        where N(+) is the number of values above mu0, N(-) is the number of values below. Values equal to mu0 are discarded.
        The p-value for M is calculated using the binomial distribution and can be interpreted the same as for a t-test. 
        The test-statistic is distributed Binom(min(N(+), N(-)), n_trials, .5) where n_trials equals N(+) + N(-).

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to compute the distribution for.
    - attribute_description (str, optional): Description of the attribute. Default is None.
    - pairs_description (str, optional): Description of the pairs. Default is None.

    Returns:
    - results_df (pandas.DataFrame): A DataFrame containing the computed distribution. The columns are: 'Attribute', 'Pairs', 'Ties5', '.05()', '.50()', '.95()', 'm()'.

    """
    median = df[column].median()
    average = df[column].mean()
    quantile_5th = df[column].quantile(0.05)
    quantile_95th = df[column].quantile(0.95)
    M, p_value = sign_test(df[column], mu0=0)
    if debug:
        print(f'[compute_distribution][column:{column}] M: {M}, p-value: {p_value}')
    

    # Compute the percentage of '{column}_diff' values between -5 and +5
    ties5 = (df[(df[column] >= -5) & (df[column] <= 5)].shape[0] / df.shape[0]) * 100

    if attribute_description is None:
        attribute_description = ''
    if pairs_description is None:
        pairs_description = ''
    
    alpha = 0.05
    # bonferroni_divisor = 9 
    # alpha_corrected = alpha / bonferroni_divisor

    if p_value < alpha:
        p_value_str = f'\\textbf{{<{alpha:.2f}}}'
    else:
        p_value_str = f'{p_value:.2f}'

    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'Attribute': attribute_description,
        'Pairs': pairs_description,
        'Ties5': f'{ties5:.0f}\\%',
        '.05()': f'{quantile_5th:.0f} €',
        '.50()': f'{median:.0f} €',
        '.95()': f'{quantile_95th:.0f} €',
        'm()': f'{average:.0f} €',
        'p-value': p_value_str
    }, index=[0])

    return results_df


def differences_distribution(df, column, test_value, baseline_value, diff_column):
    """
    Compute the distribution of differences between two groups in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name used to define the groups.
        test_value: The value representing the test group.
        baseline_value: The value representing the baseline group.
        diff_column (str): The column name containing the values to compare.

    Returns:
        pandas.DataFrame: A DataFrame containing the distribution of differences.

    Raises:
        None

    Examples:
        >>> df = pd.DataFrame({'gender': ['Male', 'Female', 'Male', 'Female'],
        ...                    'age': [25, 30, 35, 40],
        ...                    'income': [50000, 60000, 55000, 65000]})
        >>> differences_distribution(df, 'gender', 'Male', 'Female', 'income')
        
    """
    columns=['gender', 'birthplace', 'age', 'city', 'marital_status', 'education', 'profession', 'car', 'km_driven', 'class', diff_column]
    columns_features = columns.copy()
    columns_features.remove(diff_column)
    merge_on = columns.copy()
    merge_on.remove(column)
    merge_on.remove(diff_column)
    
    debug = False

    df = df[columns]

    df_base = df[df[column] == baseline_value].sort_values(by=columns_features)
    df_test = df[df[column] == test_value].sort_values(by=columns_features)
    df_merged = df_base.merge(df_test, on=merge_on, suffixes=('', '_test'))
    df_merged[f'{diff_column}_diff'] = df_merged[f'{diff_column}_test'] - df_merged[diff_column]
    
    if debug:
        print(f'df_base\n{df_base.head()}')
        print(f'df_test\n{df_test.head()}')
        print(f'df_merged\n{df_merged.head()}')
        print(f'{column} / {test_value}vs{baseline_value}: {df_base.shape} / {df_test.shape} / merged: {df_merged.shape}')
        df_sorted = df_merged.sort_values(by=f'{diff_column}_diff')
        df_sorted.to_csv(f'data/merged_data_{column}_{test_value}vs{baseline_value}.csv', sep=';', index=False)
    
    results_df = compute_distribution(df_merged, f'{diff_column}_diff', column, f'{test_value} vs {baseline_value}')
    
    return results_df