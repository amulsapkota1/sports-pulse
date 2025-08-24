import pandas as pd

def prune_missing_values(df: pd.DataFrame, row_threshold: float = 50.0) -> pd.DataFrame:
    """
    Prune DataFrame by dropping rows with missing percentage greater than row_threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        row_threshold (float): Threshold percentage of missing values per row (default 50%).

    Returns:
        pd.DataFrame: Pruned DataFrame.
    """
    # Column-level missing %
    missing_percentage_per_column = df.isnull().sum(axis=0) / len(df) * 100
    print("Missing values percentage per column:")
    print(missing_percentage_per_column)

    # Row-level missing %
    missing_percentage_per_row = df.isnull().sum(axis=1) / len(df.columns) * 100
    rows_to_drop = missing_percentage_per_row[missing_percentage_per_row > row_threshold].index

    print("\nMissing percent per row:")
    print(missing_percentage_per_row)
    print("Rows to drop (>{}% missing): {}".format(row_threshold, rows_to_drop.tolist()))

    # Drop rows
    df_pruned = df.drop(rows_to_drop)

    print("\nOriginal DataFrame shape:", df.shape)
    print("DataFrame shape after pruning:", df_pruned.shape)

    return df_pruned
