import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.inter_rater import cohens_kappa
from scipy import stats
import krippendorff

def cohens_kappa_from_df(
    df: pd.DataFrame, 
    rater_a: str, 
    rater_b: str, 
    categories: list = None,
    return_results: bool = False
):
    """
    Compute Cohen's Kappa directly from a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing ratings from two raters.
    rater_a, rater_b : str
        Column names of the two raters.
    categories : list, optional
        List of all possible categories. If None, inferred from the data.
    return_results : bool, default False
        If True, returns the full results object (including standard error).
        If False, returns only the Kappa point estimate.

    Returns:
    --------
    float or namedtuple
        Cohen's Kappa statistic (float if return_results=False).
        Full results (namedtuple with `kappa` and `std_err`) if return_results=True.

    Example:
    --------
    >>> df = pd.DataFrame({
    ...     "Rater1": [1, 2, 3, 1, 2],
    ...     "Rater2": [1, 2, 3, 1, 3]
    ... })
    >>> cohens_kappa_from_df(df, "Rater1", "Rater2", return_results=False)
    0.6363636363636364  # Just the Kappa value
    >>> cohens_kappa_from_df(df, "Rater1", "Rater2", return_results=True)
    KappaResults(kappa=0.6363636363636364, std_err=0.22188005018839622)  # Full results
    """
    # Extract ratings
    ratings_a = df[rater_a]
    ratings_b = df[rater_b]
    
    # Get unique categories if not provided
    if categories is None:
        categories = sorted(set(ratings_a).union(set(ratings_b)))
    
    # Create contingency matrix
    contingency = pd.crosstab(
        ratings_a, 
        ratings_b, 
        rownames=[rater_a], 
        colnames=[rater_b]
    ).reindex(index=categories, columns=categories, fill_value=0)
    
    # Compute Cohen's Kappa using statsmodels
    results = cohens_kappa(contingency.values)
    
    return results if return_results else results.kappa

def fleiss_kappa_from_df(df: pd.DataFrame, categories: list = None) -> float:
    """
    Compute Fleiss' Kappa directly from a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame where rows are subjects and columns are raters.
        Each cell contains the category assigned by a rater to a subject.
    categories : list, optional
        List of all possible categories. If None, inferred from the data.

    Returns:
    --------
    float
        Fleiss' Kappa statistic (between -1 and 1).

    Example:
    --------
    >>> df = pd.DataFrame({
    ...     "Rater1": [1, 2, 3, 1, 2],
    ...     "Rater2": [1, 2, 3, 1, 3],
    ...     "Rater3": [1, 2, 3, 1, 2]
    ... })
    >>> fleiss_kappa_from_df(df)
    0.7986577181208053
    """
    # Get unique categories if not provided
    if categories is None:
        categories = np.unique(df.values)
    
    n_subjects, n_raters = df.shape
    n_categories = len(categories)
    
    # Initialize an empty matrix (subjects × categories)
    agg = np.zeros((n_subjects, n_categories), dtype=int)
    
    # Count how many raters assigned each category per subject
    for i, subject in df.iterrows():
        for category_idx, category in enumerate(categories):
            agg[i, category_idx] = (subject == category).sum()
    
    # Compute Fleiss' Kappa using statsmodels
    return fleiss_kappa(agg)

def krippendorff_alpha_from_df(df, level_of_measurement='nominal'):
    """
    Computes Krippendorff's Alpha for inter-rater reliability.
    
    Parameters:
    - df: DataFrame where each column represents a rater and each row represents a subject.
    - level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio' (default: 'nominal')
    
    Returns:
    - Krippendorff's Alpha (float)
    """
    ratings = df.values.T  # Transpose to shape (n_raters, n_subjects)
    
    alpha = krippendorff.alpha(
        reliability_data=ratings,
        level_of_measurement=level_of_measurement
    )
    
    print(f"Krippendorff's Alpha ({level_of_measurement}): {alpha:.4f}")
    return alpha

def scotts_pi_from_df(df=None, rater1_col=None, rater2_col=None, return_results=False):
    """Compute Scott's Pi from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the ratings from two raters.
    rater1_col : str, optional
        Name of the column for the first rater's ratings.
    rater2_col : str, optional
        Name of the column for the second rater's ratings.
    return_results : bool, default False
        If True, returns a dict with observed agreement, expected agreement, and Scott's Pi.
        If False, returns only Scott's Pi.

    Returns
    -------
    float or dict
        Scott's Pi (if return_results=False) or full results (if return_results=True).
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns for two raters.")

    if rater1_col is None or rater2_col is None:
        ratings1 = df.iloc[:, 0]
        ratings2 = df.iloc[:, 1]
    else:
        ratings1 = df[rater1_col]
        ratings2 = df[rater2_col]

    valid_pairs = pd.DataFrame({'r1': ratings1, 'r2': ratings2}).dropna()
    n = len(valid_pairs)
    if n == 0:
        raise ValueError("No valid rating pairs after dropping missing values.")

    o_agree = np.mean(valid_pairs['r1'] == valid_pairs['r2'])

    combined = pd.concat([valid_pairs['r1'], valid_pairs['r2']])
    p = combined.value_counts(normalize=True)
    e_agree = np.sum(p**2)

    pi = 1.0 if e_agree == 1 else (o_agree - e_agree) / (1 - e_agree)

    return {'observed_agreement': o_agree,
            'expected_agreement': e_agree,
            'scotts_pi': pi} if return_results else pi

def spearman_corr_from_df(df=None, col1=None, col2=None, return_results=False):
    """
    Compute Spearman's rank correlation between two columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the two columns to compare
    col1 : str, optional
        Name of first column (default: first column in DataFrame)
    col2 : str, optional
        Name of second column (default: second column in DataFrame)
    return_results : bool, default False
        If True, returns dictionary with correlation, p-value, and sample size
        If False, returns only the correlation coefficient

    Returns
    -------
    float or dict
        Spearman's rho (if return_results=False)
        or full results (if return_results=True)
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least two columns")

    if col1 is None or col2 is None:
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
    else:
        x = df[col1]
        y = df[col2]

    valid_pairs = pd.DataFrame({'x': x, 'y': y}).dropna()
    n = len(valid_pairs)
    if n < 3:
        raise ValueError(f"Need at least 3 valid pairs (found {n})")

    rho, pval = stats.spearmanr(valid_pairs['x'], valid_pairs['y'])

    if return_results:
        return {
            'correlation': rho,
            'p_value': pval
        }
    return rho

def cronbachs_alpha_from_df(df=None, items=None, return_results=False):
    """
    Compute Cronbach's Alpha for scale reliability.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where each column represents an item in the scale
    items : list, optional
        List of column names to include (default: all numeric columns)
    return_results : bool, default False
        If True, returns dictionary with alpha, item stats, and interpretation
        If False, returns only alpha value

    Returns
    -------
    float or dict
        Cronbach's Alpha (if return_results=False)
        or full results (if return_results=True)
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None")
    
    if items is None:
        items = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(items) < 2:
        raise ValueError("Need at least 2 items to compute reliability")
    
    df_items = df[items].dropna()
    n_items = len(items)
    n_obs = len(df_items)
    
    if n_obs < 2:
        raise ValueError(f"Need at least 2 complete observations (found {n_obs})")
    
    item_vars = df_items.var(ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    
    if return_results:
        return {
            'alpha': alpha,
            'item_variance': item_vars.to_dict(),
            'scale_variance': total_var
        }
    return alpha