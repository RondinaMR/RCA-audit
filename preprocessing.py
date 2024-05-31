import pandas as pd
import numpy as np

def preprocess(df, column_prices, features, debug=False):
    """
    Preprocesses the given DataFrame by performing various data transformations.

    Args:
        df (pandas.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.

    """

    # Convert 'class' column to categorical with specified categories
    df['class'] = pd.Categorical(df['class'], ["1", "4", "9", "18"])

    # Rename C1 columns
    df.rename(columns={'C1/c': 'C1/d', 'C1/b': 'C1/c', 'C1/a': 'C1/b', 'C9': 'C1/a'}, inplace=True)

    # Create binary columns based on null values in specific columns
    df['C1'] = np.where(df[['C1/a', 'C1/b', 'C1/c', 'C1/d']].notnull().any(axis=1), 1, 0)
    df['C2'] = np.where(df[['C2/a', 'C2/b', 'C2/c']].notnull().any(axis=1), 1, 0)
    df['C3'] = np.where(df[['C3/a', 'C3/b', 'C3/c', 'C3/d']].notnull().any(axis=1), 1, 0)
    df['C4'] = np.where(df[['C4/a']].notnull().any(axis=1), 1, 0)
    df['C5'] = np.where(df[['C5/a', 'C5/b']].notnull().any(axis=1), 1, 0)
    df['C6'] = np.where(df[['C6/a']].notnull().any(axis=1), 1, 0)

    # Replace labels for visualization purposes
    df = df.replace(
        {'birthplace': {'Milan':'MI', 'Rome':'RO', 'Naples':'NA', 'China':'CN', 'Morocco':'MA'},
         'city': {'Milan':'MI', 'Naples':'NA'},
         'education': {'Master':'MSc', 'Without a qualification':'WaQ'},
         'profession': {'Employee':'Emp', 'Looking for a job':'LfaJ'},
         'marital_status': {'Married':'Mar', 'Single':'Sin', 'Widow':'Wid'}
        })

    if debug:
        print(df.dtypes)

    rows_before = df.shape[0]
    # Remove duplicates based on selected columns
    # df = df[df.duplicated(subset=features, keep='first')].sort_values(by=features)
    df = df.drop_duplicates(subset=features, keep='first')
    rows_after = df.shape[0]
    print(f'Number of rows deleted (duplicates): {rows_before - rows_after}')

    # Calculate top prices for each row
    df['top1'] = df[column_prices].min(axis=1)
    df['top2'] = df[column_prices].apply(lambda x: x.nsmallest(2).iloc[-1], axis=1)
    df['top3'] = df[column_prices].apply(lambda x: x.nsmallest(3).iloc[-1], axis=1)
    df['top4'] = df[column_prices].apply(lambda x: x.nsmallest(4).iloc[-1], axis=1)
    df['top5'] = df[column_prices].apply(lambda x: x.nsmallest(5).iloc[-1], axis=1)

    # Create new columns with selected top prices
    # df['top123'] = df.apply(lambda row: [value for value in [row['top1'], row['top2'], row['top3']] if not pd.isnull(value)], axis=1)
    df['top12345'] = df.apply(lambda row: [value for value in [row['top1'], row['top2'], row['top3'], row['top4'], row['top5']] if not pd.isnull(value)], axis=1)

    # exploded_top3_df = df.explode('top123').reset_index(drop=True)
    # exploded_top3_df = exploded_top3_df.infer_objects()
    # exploded_top3_df['class'] = pd.Categorical(exploded_top3_df['class'], ["1", "4", "9", "18"])

    # exploded_top5_df = df.explode('top12345').reset_index(drop=True)
    # exploded_top5_df = exploded_top5_df.infer_objects()
    # exploded_top5_df['class'] = pd.Categorical(exploded_top5_df['class'], ["1", "4", "9", "18"])

    # df['top5avg'] = df['top12345'].apply(lambda x: np.mean(x) if x else np.nan)
    df['top5avg'] = df['top12345'].apply(lambda x: np.mean(x))
    df = df.infer_objects()
    df['class'] = pd.Categorical(df['class'], ["1", "4", "9", "18"])
    columns_to_drop = ['top12345', 'top2', 'top3', 'top4', 'top5']
    # columns_to_drop.extend(column_prices)
    df = df.drop(columns=columns_to_drop)
    df.to_csv('debug/top_data.csv', sep=';', index=False)

    return df