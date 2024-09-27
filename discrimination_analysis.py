import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

def compute_distribution(df, column, attribute_description=None, pairs_description=None, quartiles=False, numeric=False, debug=False):
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
    - quartiles (bool, optional): Whether to compute the quartiles. Default is False.
    - numeric (bool, optional): Whether the column is numeric. Default is False.
    - debug (bool, optional): Whether to print debug information. Default is False.

    Returns:
    - results_df (pandas.DataFrame): A DataFrame containing the computed distribution. The columns are: 'Attribute', 'Pairs', 'Ties5', '.05()', '.50()', '.95()', 'm()' (quartiles = False) or 'Attribute', 'Pairs', 'Ties5', '.05()', '.25()', '.50()', '.75()', '.95()', 'm()' (quartiles = True).

    """
    median = df[column].median()
    average = df[column].mean()
    quantile_5th = df[column].quantile(0.05)
    quantile_25th = df[column].quantile(0.25)
    quantile_75th = df[column].quantile(0.75)
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
    if numeric and quartiles:
        results_df = pd.DataFrame({
            'Attribute': attribute_description,
            'Pairs': pairs_description,
            'Ties5': ties5,
            '.05()': quantile_5th,
            '.25()': quantile_25th,
            '.50()': median,
            '.75()': quantile_75th,
            '.95()': quantile_95th,
            'm()': average,
            'p-value': p_value_str
        }, index=[0])
    elif numeric and not quartiles:
        results_df = pd.DataFrame({
            'Attribute': attribute_description,
            'Pairs': pairs_description,
            'Ties5': ties5,
            '.05()': quantile_5th,
            '.50()': median,
            '.95()': quantile_95th,
            'm()': average,
            'p-value': p_value_str
        }, index=[0])
    elif not numeric and quartiles:
        results_df = pd.DataFrame({
            'Attribute': attribute_description,
            'Pairs': pairs_description,
            'Ties5': f'{ties5:.0f}\\%',
            '.05()': f'{quantile_5th:.0f} €',
            '.25()': f'{quantile_25th:.0f} €',
            '.50()': f'{median:.0f} €',
            '.75()': f'{quantile_75th:.0f} €',
            '.95()': f'{quantile_95th:.0f} €',
            'm()': f'{average:.0f} €',
            'p-value': p_value_str
        }, index=[0])
    else:
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

def create_diff_df(df, column, test_value, baseline_value, diff_column, columns_merge, debug=False):
    df_base = df[df[column] == baseline_value]
    df_test = df[df[column] == test_value]
    df_merged = df_base.merge(df_test, how='inner', on=columns_merge, suffixes=('', '_test'))
    df_merged[f'{diff_column}_diff'] = df_merged[f'{diff_column}_test'] - df_merged[diff_column]
    if debug:
        df_base.to_csv(f'debug/2_base_data_{column}_{baseline_value}.csv', sep=';', index=False)
        df_test.to_csv(f'debug/2_test_data_{column}_{test_value}.csv', sep=';', index=False)
        print(f'{column} / {test_value}vs{baseline_value}: {df_base.shape} / {df_test.shape} / merged: {df_merged.shape}')
    return df_merged


def differences_distribution(df, column, test_value, baseline_value, diff_column, quartiles=False, numeric=False, debug=False):
    """
    Compute the distribution of differences between two groups in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        column (str): The column name used to define the groups.
        test_value: The value representing the test group.
        baseline_value: The value representing the baseline group.
        diff_column (str): The column name containing the values to compare.
        quartiles (bool, optional): Whether to compute the quartiles. Defaults to False.
        debug (bool, optional): Whether to print debug information. Defaults to False.

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

    df_original = df[columns]

    df = df_original.drop_duplicates(subset=columns_features)
    num_rows_deleted = df_original.shape[0] - df.shape[0]
    if debug:
        print(f'Number of rows deleted: {num_rows_deleted}')

    df_merged = create_diff_df(df, column, test_value, baseline_value, diff_column, merge_on, debug=debug)
    
    if debug:
        df_sorted = df_merged.sort_values(by=f'{diff_column}_diff')
        df_sorted.to_csv(f'debug/2_merged_data_{column}_{test_value}vs{baseline_value}.csv', sep=';', index=False)
    
    results_df = compute_distribution(df_merged, f'{diff_column}_diff', column, f'{test_value} vs {baseline_value}', quartiles=quartiles, numeric=numeric, debug=debug)
    
    # TODO: Perform t-test on the differences
    # https://stackoverflow.com/questions/59694680/how-do-i-perform-a-t-test-from-a-dataframe
    # var_base = df_base["hourly_wage"].to_numpy()
    # var_test = df_test["hourly_wage"].to_numpy()
    # stats.ttest_ind(m,f)
    #compute p-value
    # print(df_merged.head())
    # print(stats.ttest_ind(df_merged[f'{diff_column}'].to_numpy(),df_merged[f'{diff_column}_test'].to_numpy()))

    return results_df

def control_pairs(df_original, df_cp, features, column_name, debug=False):
    """
    Compute control pairs for a given DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        features (list): List of column names to consider for identifying duplicates.
        column_name (str, optional): The column name to compute the difference for. Defaults to 'top1'.

    Returns:
        DataFrame: The computed control pairs.

    """
    # Use the original Dataframe, find duplicates, and use them as control pairs
    # df = df_original.copy()
    # df_copy = df[df.duplicated(subset=features, keep='first')].sort_values(by=features)
    # if debug:
    #     print(f'#duplicates: {df.shape[0]}')
    #     df.to_csv(f'debug/2_duplicates_{column_name}.csv', sep=';', index=False)
    # df = df.merge(df_copy, how='inner', on=features, suffixes=('', '_cp'))
    # df[f'{column_name}_diff'] = df[f'{column_name}_cp'] - df[column_name]
    # # df[f'{column_name}_diff'] = df.groupby(features, observed=True)[column_name].transform(lambda x: x.diff())
    # # df = df.dropna(subset=[f'{column_name}_diff'])
    # cp1 = compute_distribution(df, f'{column_name}_diff', 'control pairs 1')
    # if debug:
    #     print(f'#control_pairs: {df.shape[0]}')
    #     df.to_csv(f'debug/2_control_pairs-1_{column_name}.csv', sep=';', index=False)


    # Merge the original DataFrame with the control pairs DataFrame
    df = df_original.copy()
    df = df.merge(df_cp, how='inner', on=features, suffixes=('', '_cp'))
    df[f'{column_name}_diff'] = df[f'{column_name}_cp'] - df[column_name]
    cp = compute_distribution(df, f'{column_name}_diff', 'control pairs')

    if debug:
        print(f'#control_pairs: {df.shape[0]}')
        df.to_csv(f'debug/2_control_pairs_{column_name}.csv', sep=';', index=False)
    
    return cp