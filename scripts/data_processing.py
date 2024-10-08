import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import OneHotEncoder

# Define the function to analyze the numeric column
def analyze_numeric_column(df: pd.DataFrame, column_name: str) -> None:
    """
    Analyze a numeric column in a DataFrame, providing summary statistics, unique values,
    frequency distribution, and visualizations.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The numeric column to be analyzed.
    """
    print(f"Analyzing numeric column: {column_name}\n")
    print("Data type:", df[column_name].dtype)

    # Number of unique values
    unique_values = df[column_name].nunique()
    print("Number of unique values:", unique_values)

    # Combined histogram for all clients and subscribed clients
    plt.figure(figsize=(12, 6))
    plt.hist(df[column_name], bins=30, edgecolor='black', alpha=0.6, color='#28698a', label='All clients')
    plt.hist(df[df['y'] == 'yes'][column_name], bins=30, edgecolor='black', alpha=0.6, color='#529943', label='Subscribed clients')

    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Binning numeric columns for better visualization
    bins = np.histogram_bin_edges(df[column_name], bins='auto')
    if len(bins) > 20:
        bins = np.linspace(df[column_name].min(), df[column_name].max(), 21)
    df['binned_numeric'] = pd.cut(df[column_name], bins)
    df_grouped = df.groupby('binned_numeric', observed=False)['y'].apply(lambda x: (x == 'yes').mean() * 100)

    plt.figure(figsize=(15, 7))
    df_grouped.plot(kind='bar', color='#529943', edgecolor='black', alpha=0.7)
    plt.xlabel(f'Binned {column_name}')
    plt.ylabel('Percentage of clients who subscribed (%)')
    plt.title(f'Percentage of subscriptions within each {column_name} group')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Drop temporary columns
    df.drop(columns=['binned_numeric'], inplace=True)

# Define the function to analyze the categorical column
def analyze_categorical_column(df: pd.DataFrame, column_name: str) -> None:
    """
    Analyze a categorical column in a DataFrame, providing summary statistics, unique values,
    frequency distribution, and visualizations.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The categorical column to be analyzed.
    """
    print(f"Analyzing categorical column: {column_name}\n")
    print("Data type:", df[column_name].dtype)

    # Number of unique values
    unique_values = df[column_name].nunique()
    print("Number of unique values:", unique_values)

    # Display unique values
    print("\nUnique values:")
    print(df[column_name].unique())

    # Value counts
    print("\nValue counts:")
    print(df[column_name].value_counts())

    # Sort months if column is 'month'
    if column_name == 'month':
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        df[column_name] = pd.Categorical(df[column_name].str.lower(), categories=month_order, ordered=True)

    # Combined bar plot for all clients and subscribed clients
    total_counts = df[column_name].value_counts().sort_index()
    subscribed_counts = df[df['y'] == 'yes'][column_name].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(total_counts.index, total_counts.values, color='#28698a', alpha=0.6, label='All clients', edgecolor='black')
    plt.bar(subscribed_counts.index, subscribed_counts.values, color='#529943', alpha=0.6, label='Subscribed clients', edgecolor='black')

    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.title(f'Frequency distribution of {column_name}')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Percentage distribution of subscription within each group
    df_grouped = df.groupby(column_name, observed=False)['y'].apply(lambda x: (x == 'yes').mean() * 100)

    plt.figure(figsize=(15, 7))
    df_grouped.plot(kind='bar', color='#529943', edgecolor='black', alpha=0.7)
    plt.xlabel(column_name)
    plt.ylabel('Percentage of clients who subscribed (%)')
    plt.title(f'Percentage of subscriptions within each group for {column_name}')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Define the function to analyze the specified column
def analyze_column(df: pd.DataFrame, column_name: str) -> None:
    """
    Analyze a specified column in a DataFrame, routing to appropriate analysis function based on data type.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The column to be analyzed.
    """
    # Check if column exists in DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame.")
        return

    if pd.api.types.is_numeric_dtype(df[column_name]):
        analyze_numeric_column(df, column_name)
    elif isinstance(df[column_name].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[column_name]):
        analyze_categorical_column(df, column_name)
    else:
        print(f"Column '{column_name}' has unsupported data type for analysis.")

# Define the function to analyze the target column
def analyze_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    Analyze the target column in a DataFrame, providing the percentage distribution
    to determine if it is balanced or not.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The target column to be analyzed.

    Returns:
    None: This function outputs a visual representation and prints the percentage distribution.
    """
    # Calculate value counts and percentage distribution
    value_counts: pd.Series = df[target_column].value_counts()
    percentage_distribution: pd.Series = (value_counts / len(df)) * 100

    # Print the percentage distribution of each unique value
    print(f"Percentage distribution of '{target_column}':")
    for value, percentage in percentage_distribution.items():
        print(f"{value}: {percentage:.2f}%")

    # Visualize the percentage distribution using a bar plot
    plt.figure(figsize=(10, 6))
    percentage_distribution.plot(kind='bar', color='#28698a', edgecolor='black', alpha=0.7)
    plt.xlabel(target_column)
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage distribution of target column: {target_column}')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Encode categorical columns
def encode_categorical_columns(df: pd.DataFrame, obj_columns: List[str], binary_columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical columns in the DataFrame using one-hot encoding for non-binary columns,
    ordinal encoding for columns with a natural order, and binary mapping for binary columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    obj_columns (List[str]): List of non-binary categorical columns to be encoded.
    binary_columns (List[str]): List of binary categorical columns to be mapped.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    # One-hot encode non-binary categorical columns, excluding 'month' and 'education'
    non_binary_columns = [col for col in obj_columns if col not in ['month', 'education']]
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_arrays = encoder.fit_transform(df[non_binary_columns])
    encoded_df = pd.DataFrame(encoded_arrays, columns=encoder.get_feature_names_out(non_binary_columns))

    # Map 'month' column to numeric values (1-12)
    month_mapping: dict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    if 'month' in df.columns:
        df['month'] = df['month'].map(month_mapping)

    # Map 'education' column to ordinal values
    education_mapping: dict = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    if 'education' in df.columns:
        df['education'] = df['education'].map(education_mapping)

    # Drop original non-binary categorical columns and concatenate encoded columns
    df = df.drop(columns=obj_columns).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    # Map binary categorical columns
    for col in binary_columns:
        if col in df.columns:
            if df[col].isin(['yes', 'no']).all():
                df[col] = df[col].map({'yes': 1, 'no': 0})
            else:
                raise ValueError(f"Column '{col}' contains values other than 'yes' or 'no'. Please verify your data.")
        else:
            raise KeyError(f"Column '{col}' is not found in the DataFrame. Please check your column list.")

    return df

# Detect outliers using IQR and Z-score methods
def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> Dict[str, pd.DataFrame]:
    """
    Detect outliers in the specified columns of a DataFrame using the selected method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (List[str]): List of columns to check for outliers.
    method (str): Method to use for detecting outliers ('iqr' or 'zscore'). Defaults to 'iqr'.
    threshold (float): Threshold for detecting outliers (1.5 for IQR or 3 for Z-score). Defaults to 1.5.

    Returns:
    Dict[str, pd.DataFrame]: A dictionary where keys are column names and values are DataFrames containing the outliers.
    """
    # Dictionary to store outlier DataFrames
    outliers: Dict[str, pd.DataFrame] = {}

    # Check that the method is valid
    if method not in ['iqr', 'zscore']:
        raise ValueError("The 'method' parameter must be either 'iqr' or 'zscore'.")

    # Detect outliers for each column in the list
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in the DataFrame.")

        if method == 'iqr':
            # Interquartile Range (IQR) method
            Q1: float = df[column].quantile(0.25)
            Q3: float = df[column].quantile(0.75)
            IQR: float = Q3 - Q1
            lower_bound: float = Q1 - threshold * IQR
            upper_bound: float = Q3 + threshold * IQR
            outlier_df: pd.DataFrame = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        elif method == 'zscore':
            # Z-score method
            mean: float = df[column].mean()
            std: float = df[column].std()
            z_scores: pd.Series = (df[column] - mean) / std
            outlier_df: pd.DataFrame = df[z_scores.abs() > threshold]

        # Store non-empty outlier DataFrames in the dictionary
        if not outlier_df.empty:
            outliers[column] = outlier_df

    return outliers

# Handling outliers
def handle_outliers(df: pd.DataFrame, outliers: Dict[str, pd.DataFrame], method: str = 'remove', fill_value: Any = None) -> pd.DataFrame:
    """
    Handle outliers in the DataFrame using the specified method.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    outliers (Dict[str, pd.DataFrame]): A dictionary of outliers returned by `detect_outliers()` where keys are column names and values are DataFrames of outliers.
    method (str): Method to handle outliers ('remove', 'cap', 'fill'). Defaults to 'remove'.
    fill_value (Any): Value to use when filling outliers (used if method is 'fill').

    Returns:
    pd.DataFrame: The DataFrame with handled outliers.
    """
    # Check for valid method input
    if method not in ['remove', 'cap', 'fill']:
        raise ValueError("The 'method' parameter must be either 'remove', 'cap', or 'fill'.")

    # Handle outliers for each specified column
    for column, outlier_df in outliers.items():
        if column not in df.columns:
            raise KeyError(f"Column '{column}' is not present in the DataFrame.")

        if method == 'remove':
            # Remove outlier rows from the DataFrame
            df = df.drop(outlier_df.index).reset_index(drop=True)
        elif method == 'cap':
            # Cap outliers to the IQR bounds
            Q1: float = df[column].quantile(0.25)
            Q3: float = df[column].quantile(0.75)
            IQR: float = Q3 - Q1
            lower_bound: float = Q1 - 1.5 * IQR
            upper_bound: float = Q3 + 1.5 * IQR
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        elif method == 'fill':
            # Replace outliers with the specified fill value
            if fill_value is None:
                raise ValueError("The 'fill_value' parameter must be specified when using the 'fill' method.")
            df.loc[outlier_df.index, column] = fill_value

    return df

# Calculate percentage of outliers
def calculate_outlier_percentage(outliers: Dict[str, pd.DataFrame], total_rows: int) -> float:
    """
    Calculate and print the percentage of outliers in the DataFrame.

    Parameters:
    outliers (Dict[str, pd.DataFrame]): A dictionary of outliers returned by `detect_outliers()`, 
                                        where keys are column names and values are DataFrames containing outlier rows.
    total_rows (int): Total number of rows in the original DataFrame.

    Returns:
    float: The percentage of data points that are outliers.
    """
    # Validate input parameters
    if not isinstance(total_rows, int) or total_rows <= 0:
        raise ValueError("The 'total_rows' parameter must be a positive integer.")
    
    # Calculate the total number of outliers across all specified columns
    total_outliers: int = sum(len(outlier_df) for outlier_df in outliers.values())
    
    # Calculate the percentage of outliers
    percentage_outliers: float = (total_outliers / total_rows) * 100
    
    # Print the percentage of outliers
    print(f"Percentage of data that are outliers: {round(percentage_outliers, 2)}%")
    
    return round(percentage_outliers, 2)