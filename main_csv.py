import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, bartlett, fligner, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from typing import Tuple
import re
import numpy as np


def load_cytokine_data(file_path: str) -> pd.DataFrame:
    """
    Loads cytokine data from the CSV file.
    """
    try:
        cytokine_df = pd.read_csv(file_path)
        logging.info(f"Cytokine data loaded with shape {cytokine_df.shape}.")
        return cytokine_df
    except FileNotFoundError:
        logging.error(f"Cytokine file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading cytokine data: {e}")
        return pd.DataFrame()


def filter_cytokine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters cytokine data by removing rows with missing patient information.
    """
    required_columns = ['Patient ID', 'Control/Participant', 'Sex', 'Age', 'COVID']
    missing_info = df[required_columns].isnull().any(axis=1)
    filtered_df = df[~missing_info]
    logging.info(f"Cytokine data filtered from {df.shape[0]} to {filtered_df.shape[0]} rows.")
    return filtered_df


def get_cytokine_markers(columns):
    pattern = re.compile(r'.*\(\d+\)$')  # Adjust this pattern based on your column naming
    return [col for col in columns if pattern.match(col)]


def remove_high_oor_markers(df: pd.DataFrame, oor_value: float = -1.0, threshold: float = 0.8) -> pd.DataFrame:
    """
    Removes cytokine marker columns where more than a specified threshold of values are OOR.
    """
    # Identify cytokine marker columns
    markers = get_cytokine_markers(df.columns)

    # Calculate the proportion of OOR values for each marker
    oor_proportions = df[markers].eq(oor_value).mean()

    # Identify markers exceeding the OOR threshold
    markers_to_drop = oor_proportions[oor_proportions > threshold].index.tolist()

    if markers_to_drop:
        df = df.drop(columns=markers_to_drop)
        logging.info(f"Dropped {len(markers_to_drop)} markers due to >{threshold * 100}% OOR values: {markers_to_drop}")
    else:
        logging.info(f"No markers exceeded the {threshold * 100}% OOR threshold.")

    return df


def impute_missing_values(df: pd.DataFrame, oor_value: float = -1.0, median_threshold: float = 0.2,
                          knn_threshold: float = 0.8) -> pd.DataFrame:
    """
    Imputes missing values in cytokine markers using median or KNN imputation based on the proportion of missingness.
    """
    # Identify cytokine marker columns
    markers = get_cytokine_markers(df.columns)

    # Replace OORs with NaN
    df[markers] = df[markers].replace(oor_value, np.nan)

    # Calculate missingness per marker
    missing_proportions = df[markers].isna().mean()

    # Split markers based on missingness
    markers_median = missing_proportions[missing_proportions <= median_threshold].index.tolist()
    markers_knn = missing_proportions[
        (missing_proportions > median_threshold) & (missing_proportions <= knn_threshold)].index.tolist()

    logging.info(f"{len(markers_median)} markers with <=20% missingness.")
    logging.info(f"{len(markers_knn)} markers with >20% and <=80% missingness.")

    # Median Imputation for markers with <=20% missingness
    for col in markers_median:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        logging.debug(f"Imputed median for {col}: {median}")

    # KNN Imputation for markers with >20% and <=80% missingness
    if markers_knn:
        imputer = KNNImputer(n_neighbors=5)
        # Apply KNN imputation only on markers_knn
        impute_df = df[markers_knn]
        imputed_values = imputer.fit_transform(impute_df)
        imputed_df = pd.DataFrame(imputed_values, columns=markers_knn, index=df.index)
        df[markers_knn] = imputed_df
        logging.info(f"Imputed KNN for {len(markers_knn)} markers.")

    return df


def convert_markers_to_numeric(df: pd.DataFrame, value_vars: list) -> pd.DataFrame:
    """
    Converts specified columns to numeric, coercing errors to NaN.
    """
    for col in value_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    logging.info("Converted cytokine marker columns to numeric types.")
    return df


def clean_and_impute_cytokine_data_wide(df: pd.DataFrame, oor_value: float = -1.0, median_threshold: float = 0.2,
                                        knn_threshold: float = 0.8) -> pd.DataFrame:
    """
    Cleans and imputes cytokine data in wide format.
    """
    # Step 1: Remove markers with >80% OOR values
    df = remove_high_oor_markers(df, oor_value=oor_value, threshold=knn_threshold)

    # Identify cytokine marker columns
    value_vars = get_cytokine_markers(df.columns)

    # Step 2: Impute missing values
    df = impute_missing_values(df, oor_value=oor_value, median_threshold=median_threshold, knn_threshold=knn_threshold)

    # Step 3: Convert marker columns to numeric
    df = convert_markers_to_numeric(df, value_vars)

    logging.info(f"Data cleaning and imputation completed on wide-format data.")
    return df


def extract_cytokine_groups(df: pd.DataFrame) -> dict:
    """
    Extracts specific groups from the cytokine DataFrame.
    """
    df['Group'] = df.apply(
        lambda row: 'Control_Negative' if row['Control/Participant'] == 'C' and row['COVID'] == 'N' else
        'Participant_Positive' if row['Control/Participant'] == 'P' and row['COVID'] == 'Y' else None,
        axis=1
    )
    groups = {
        'Control_Negative': df[df['Group'] == 'Control_Negative'],
        'Participant_Positive': df[df['Group'] == 'Participant_Positive'],
    }
    for label, group_df in groups.items():
        logging.info(f"{label} group has {group_df.shape[0]} patients.")
    return groups


def reshape_cytokine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes cytokine data from wide to long format.
    """
    id_vars = ['Patient ID', 'Control/Participant', 'Sex', 'Age', 'COVID', 'Group']
    value_vars = get_cytokine_markers(df.columns)

    # Ensure that all necessary columns are present
    for col in id_vars:
        if col not in df.columns:
            logging.error(f"Column '{col}' is missing from the DataFrame.")
            return pd.DataFrame()

    # Before melt
    logging.info(f"Before melt: {df.shape[0]} rows, {df.shape[1]} columns")
    logging.debug(f"Sample data before melt:\n{df.head()}")

    # Melt the DataFrame
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                        var_name='Cytokine_Marker', value_name='Value')

    # After melt
    logging.info(f"After melt: {melted_df.shape[0]} rows, {melted_df.shape[1]} columns")
    logging.debug(f"Sample data after melt:\n{melted_df.head(10)}")

    return melted_df


def compare_cytokine_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, group1_label: str,
                            group2_label: str, significance_level: float = 0.05) -> pd.DataFrame:
    """
    Compares cytokine markers between two groups using raw values.
    """
    markers = group1_df['Cytokine_Marker'].unique()
    results = []

    for marker in markers:
        group1_values = group1_df[group1_df['Cytokine_Marker'] == marker]['Value'].dropna()
        group2_values = group2_df[group2_df['Cytokine_Marker'] == marker]['Value'].dropna()

        if len(group1_values) > 1 and len(group2_values) > 1:

            # Normality test
            shapiro_p1 = shapiro(group1_values)[1]
            shapiro_p2 = shapiro(group2_values)[1]
            normal = (shapiro_p1 >= significance_level) and (shapiro_p2 >= significance_level)

            # Homoscedasticity test
            if normal:
                # Data is normal; use Bartlett’s Test
                levene_p = bartlett(group1_values, group2_values)[1]
                homoscedastic = (levene_p >= significance_level)
                homoscedasticity_test = 'Bartlett’s Test'
            else:
                # Data not normal; use Levene’s Test with median
                levene_p = levene(group1_values, group2_values, center='median')[1]
                homoscedastic = (levene_p >= significance_level)
                homoscedasticity_test = 'Levene’s Test (median)'

            # Decide which test to use based on assumptions
            if normal and homoscedastic:
                # Use standard t-test
                t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=True)
                test_used = 't-test (equal variances)'
            elif normal and not homoscedastic:
                # Use Welch's t-test
                t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
                test_used = 'Welch\'s t-test (unequal variances)'
            else:
                # Use Mann-Whitney U Test
                t_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                test_used = 'Mann-Whitney U Test (non-parametric)'

            result = {
                'Comparison': f'{group1_label} vs {group2_label}',
                'Cytokine_Marker': marker,
                f'{group1_label}_Mean': group1_values.mean(),
                f'{group2_label}_Mean': group2_values.mean(),
                'Test_Used': test_used,
                'Statistic': t_stat,
                'Raw_P_Value': p_value,
                'Normality_Group1_P': shapiro_p1,
                'Normality_Group2_P': shapiro_p2,
                'Homoscedasticity_Test': homoscedasticity_test,
                'Homoscedasticity_P': levene_p,
                'Assumption_Normality': normal,
                'Assumption_Homoscedasticity': homoscedastic
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values('Raw_P_Value', inplace=True)
    results_df = adjust_p_values(results_df, p_value_column='Raw_P_Value', significance_level=significance_level)
    results_df.to_csv('cytokine_comparison_results.csv', index=False)
    logging.info("Cytokine Comparison Results saved to 'cytokine_comparison_results.csv'.")

    return results_df


def normalize_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalizes specified columns in the DataFrame using Z-score normalization.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logging.info("Data normalization (Z-score) completed.")
    return df


def adjust_p_values(results_df: pd.DataFrame, p_value_column: str = 'P_Value',
                    significance_level: float = 0.05) -> pd.DataFrame:
    """
    Adjusts p-values for multiple comparisons using the Benjamini-Hochberg procedure
    and adds significance flags based on raw and adjusted p-values.
    """
    if p_value_column not in results_df.columns:
        logging.error(f"{p_value_column} column not found in results DataFrame.")
        return results_df

    p_values = results_df[p_value_column]
    adjusted = multipletests(p_values, method='fdr_bh')
    results_df['Adjusted_P_Value'] = adjusted[1]
    results_df['Significant_Adjusted'] = adjusted[0]

    # Add raw significance flag
    results_df['Significant_Raw'] = results_df[p_value_column] < significance_level

    return results_df


def perform_pca(df: pd.DataFrame, n_components: int = 2, group_labels: pd.Series = None, title: str = "PCA") -> Tuple[
    PCA, pd.DataFrame]:
    """
    Performs PCA on the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input data.
        n_components (int): Number of principal components to compute.
        group_labels (pd.Series): Optional series indicating group membership for coloring.
        title (str): Title for the PCA plot.

    Returns:
        Tuple[PCA, pd.DataFrame]: The PCA object and the transformed DataFrame.
    """
    if df.empty:
        logging.warning("Input DataFrame for PCA is empty. Skipping PCA.")
        return None, pd.DataFrame()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Initialize PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)],
                          index=df.index)

    if group_labels is not None:
        pca_df = pca_df.join(group_labels)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='PC1', y='PC2',
            hue='Group',
            palette='Set1',
            data=pca_df.reset_index(),
            alpha=0.7
        )
        plt.title(title)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)')
        plt.legend(title='Group')
        plt.tight_layout()
        plt.savefig(f'{title}_PCA.png')
        plt.close()

    logging.info(f"PCA completed for {title}. Explained variance: {pca.explained_variance_ratio_}")

    return pca, pca_df


def load_autoantibody_data(file_path: str) -> pd.DataFrame:
    """
    Loads autoantibody data from the CSV file.
    """
    try:
        autoantibody_df = pd.read_csv(file_path)
        logging.info(f"Autoantibody data loaded with shape {autoantibody_df.shape}.")
        return autoantibody_df
    except FileNotFoundError:
        logging.error(f"Autoantibody file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading autoantibody data: {e}")
        return pd.DataFrame()


def extract_autoantibody_groups(df: pd.DataFrame, cytokine_df: pd.DataFrame) -> dict:
    """
    Extracts specific groups from the autoantibody DataFrame.
    """
    # Transpose the DataFrame to get patient IDs
    patients = df.columns[2:]
    patient_info = pd.DataFrame({'Patient ID': patients})

    # Merge with group labels from cytokine data
    group_labels = cytokine_df[['Patient ID', 'Control/Participant', 'COVID']].drop_duplicates()
    patient_info = patient_info.merge(group_labels, on='Patient ID', how='left')

    groups = {
        'Control_Negative': patient_info[
            (patient_info['Control/Participant'] == 'C') & (patient_info['COVID'] == 'N')
            ]['Patient ID'].tolist(),
        'Participant_Positive': patient_info[
            (patient_info['Control/Participant'] == 'P ') & (patient_info['COVID'] == 'Y')
            ]['Patient ID'].tolist(),
    }
    for label, patients in groups.items():
        logging.info(f"{label} group has {len(patients)} patients.")
    return groups


def reshape_autoantibody_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes autoantibody data from wide to long format.
    """
    id_vars = ['Marker', 'Immunoglobulin']
    value_vars = [col for col in df.columns if col not in id_vars]
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                        var_name='Patient ID', value_name='Value')
    logging.info(f"Autoantibody data reshaped to long format with shape {melted_df.shape}.")
    return melted_df


def compare_autoantibody_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, group1_label: str,
                                group2_label: str) -> pd.DataFrame:
    """
    Compares autoantibody markers between two groups.
    """
    combined_df = pd.concat([
        group1_df.assign(Group=group1_label),
        group2_df.assign(Group=group2_label)
    ])

    combinations_unique = combined_df[['Immunoglobulin', 'Marker']].drop_duplicates()
    results = []

    for _, row in combinations_unique.iterrows():
        ig_type = row['Immunoglobulin']
        marker = row['Marker']
        marker_data = combined_df[
            (combined_df['Immunoglobulin'] == ig_type) &
            (combined_df['Marker'] == marker)
            ]
        group1_values = marker_data[marker_data['Group'] == group1_label]['Z_Score'].dropna()
        group2_values = marker_data[marker_data['Group'] == group2_label]['Z_Score'].dropna()

        if len(group1_values) > 1 and len(group2_values) > 1:
            t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
            result = {
                'Comparison': f'{group1_label} vs {group2_label}',
                'Immunoglobulin': ig_type,
                'Marker': marker,
                f'{group1_label}_Mean_Z': group1_values.mean(),
                f'{group2_label}_Mean_Z': group2_values.mean(),
                'T_Statistic': t_stat,
                'P_Value': p_value
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values('P_Value', inplace=True)
    results_df = adjust_p_values(results_df)
    results_df.to_csv('autoantibody_comparison_results.csv', index=False)
    logging.info("Autoantibody Comparison Results:\n%s", results_df.head())

    return results_df


def adjust_p_values(results_df: pd.DataFrame, p_value_column: str = 'P_Value',
                    significance_level: float = 0.05) -> pd.DataFrame:
    """
    Adjusts p-values for multiple comparisons using the Benjamini-Hochberg procedure
    and adds significance flags based on raw and adjusted p-values.
    """
    if p_value_column not in results_df.columns:
        logging.error(f"{p_value_column} column not found in results DataFrame.")
        return results_df

    p_values = results_df[p_value_column]
    adjusted = multipletests(p_values, method='fdr_bh')
    results_df['Adjusted_P_Value'] = adjusted[1]
    results_df['Significant_Adjusted'] = adjusted[0]

    # Add raw significance flag
    results_df['Significant_Raw'] = results_df[p_value_column] < significance_level

    return results_df


def perform_pca(df: pd.DataFrame, n_components: int = 2, group_labels: pd.Series = None, title: str = "PCA") -> Tuple[
    PCA, pd.DataFrame]:
    """
    Performs PCA on the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input data.
        n_components (int): Number of principal components to compute.
        group_labels (pd.Series): Optional series indicating group membership for coloring.
        title (str): Title for the PCA plot.

    Returns:
        Tuple[PCA, pd.DataFrame]: The PCA object and the transformed DataFrame.
    """
    if df.empty:
        logging.warning("Input DataFrame for PCA is empty. Skipping PCA.")
        return None, pd.DataFrame()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Initialize PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)],
                          index=df.index)

    if group_labels is not None:
        pca_df = pca_df.join(group_labels)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='PC1', y='PC2',
            hue='Group',
            palette='Set1',
            data=pca_df.reset_index(),
            alpha=0.7
        )
        plt.title(title)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)')
        plt.legend(title='Group')
        plt.tight_layout()
        plt.savefig(f'{title}_PCA.png')
        plt.close()

    logging.info(f"PCA completed for {title}. Explained variance: {pca.explained_variance_ratio_}")

    return pca, pca_df


def main():
    """
    Main function to process the data, perform comparisons, and execute PCA.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process immunology data.')
    parser.add_argument('cytokine_file', type=str, help='Path to the cytokine CSV file')
    parser.add_argument('autoantibody_file', type=str, help='Path to the autoantibody CSV file')
    args = parser.parse_args()

    # Set up logging in INFO mode
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load and filter cytokine data
    cytokine_df = load_cytokine_data(args.cytokine_file)
    cytokine_df = filter_cytokine_data(cytokine_df)
    autoantibody_df = load_autoantibody_data(args.autoantibody_file)

    # Clean and impute data in wide format
    cytokine_df = clean_and_impute_cytokine_data_wide(cytokine_df)

    # Extract groups
    cytokine_groups = extract_cytokine_groups(cytokine_df)
    autoantibody_groups = extract_autoantibody_groups(autoantibody_df, cytokine_df)

    # Define comparisons
    comparisons = [
        ('Control_Negative', 'Participant_Positive')
    ]

    # --- Cytokine Data Processing ---
    logging.info("Starting Cytokine Data Processing...")

    for group1_label, group2_label in comparisons:
        group1_df = cytokine_groups[group1_label]
        group2_df = cytokine_groups[group2_label]

        if not group1_df.empty and not group2_df.empty:
            # Log number of samples in each group
            logging.info(f"Number of {group1_label} samples: {group1_df.shape[0]}")
            logging.info(f"Number of {group2_label} samples: {group2_df.shape[0]}")

            # Combine data for PCA
            combined_cytokine_df = pd.concat([group1_df, group2_df])

            # Extract cytokine marker columns
            value_vars = get_cytokine_markers(combined_cytokine_df.columns)

            # Normalize the data
            combined_cytokine_df = normalize_data(combined_cytokine_df, value_vars)

            # Prepare data for PCA
            pca_data = combined_cytokine_df.set_index('Patient ID')[value_vars]

            # Extract group labels
            group_labels = combined_cytokine_df.set_index('Patient ID')['Group']

            # Perform PCA
            pca_cytokine, pca_cytokine_df = perform_pca(
                df=pca_data,
                n_components=2,
                group_labels=group_labels,
                title='PCA_Cytokine'
            )
            pca_cytokine_df.to_csv('PCA_Cytokine.csv', index=False)
            logging.info("PCA results saved to 'PCA_Cytokine.csv'.")

            # Reshape data for statistical tests
            group1_long_df = reshape_cytokine_data(group1_df)
            group2_long_df = reshape_cytokine_data(group2_df)

            # Perform statistical comparison
            compare_cytokine_groups(group1_long_df, group2_long_df, group1_label, group2_label)
        else:
            logging.warning(
                f"One of the cytokine groups '{group1_label}' or '{group2_label}' is missing. Skipping Cytokine Comparisons.")

    '''
    # --- Autoantibody Data Processing ---
    logging.info("Starting Autoantibody Data Processing...")

    autoantibody_results = []
    reshaped_auto_df = reshape_autoantibody_data(autoantibody_df)
    normalized_auto_df = normalize_autoantibody_data(reshaped_auto_df)

    # Assign group labels
    normalized_auto_df['Group'] = normalized_auto_df['Patient ID'].map(
        {pid: 'Control_Negative' if pid in autoantibody_groups['Control_Negative'] else
        'Participant_Positive' if pid in autoantibody_groups['Participant_Positive'] else 'Unknown'
         for pid in normalized_auto_df['Patient ID']}
    )

    # Filter out patients with 'Unknown' group
    normalized_auto_df = normalized_auto_df[normalized_auto_df['Group'] != 'Unknown']

    for group1_label, group2_label in comparisons:
        group1_df = normalized_auto_df[normalized_auto_df['Group'] == group1_label]
        group2_df = normalized_auto_df[normalized_auto_df['Group'] == group2_label]

        if group1_df.empty or group2_df.empty:
            logging.warning(
                f"One of the autoantibody groups '{group1_label}' or '{group2_label}' is empty. Skipping comparison.")
            continue

        # Perform statistical comparison
        result_df = compare_autoantibody_groups(group1_df, group2_df, group1_label, group2_label)
        autoantibody_results.append(result_df)

    if autoantibody_results:
        combined_auto_results = pd.concat(autoantibody_results, ignore_index=True)
        combined_auto_results.to_csv('autoantibody_comparison_results.csv', index=False)
        logging.info("All Autoantibody Comparisons Completed.")

        # Perform PCA for each immunoglobulin type
        for ig_type in normalized_auto_df['Immunoglobulin'].unique():
            ig_data = normalized_auto_df[normalized_auto_df['Immunoglobulin'] == ig_type]
            pivot_df = ig_data.pivot_table(
                index='Patient ID', columns='Marker', values='Z_Score', aggfunc='mean'
            ).dropna()
            group_labels = ig_data[['Patient ID', 'Group']].drop_duplicates().set_index('Patient ID')['Group']
            group_labels = group_labels.loc[pivot_df.index]

            # Perform PCA
            pca, pca_df = perform_pca(
                df=pivot_df,
                n_components=2,
                group_labels=group_labels,
                title=f'PCA_Autoantibody_{ig_type}'
            )
            pca_df.to_csv(f'PCA_Autoantibody_{ig_type}.csv', index=False)
    else:
        logging.warning("No Autoantibody Comparisons were performed.")
    '''


if __name__ == '__main__':
    main()
