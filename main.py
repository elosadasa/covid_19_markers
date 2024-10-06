import pandas as pd
import json
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, ttest_ind
from typing import Dict, Tuple
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def filter_individuals(data: Dict) -> Dict:
    """
    Filters individuals who have both cytokine and autoantibody data,
    as well as required demographic information.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        Dict: A dictionary of filtered individuals.
    """
    filtered_data = {}
    for participant, content in data.items():
        # Check if both cytokine and autoantibody data are available
        has_cytokine = 'cytokine_data' in content and bool(content['cytokine_data'])
        has_autoantibody = 'autoantibody_data' in content and bool(content['autoantibody_data'])
        # Check for Control/Participant status and COVID self-report in demographic data
        participant_info = content.get('participant_info', {})
        control_participant_status = participant_info.get('Control/Participant')
        covid_self_report = participant_info.get('COVID self report')

        if all([
            has_cytokine,
            has_autoantibody,
            control_participant_status is not None,
            covid_self_report is not None
        ]):
            filtered_data[participant] = content

    # Log the count of individuals who meet the criteria
    count = len(filtered_data)
    logging.info(f'{count} individuals meet the criteria.')
    return filtered_data


def extract_specific_groups(data: Dict) -> Dict[str, Dict]:
    """
    Extracts the specific groups for comparison:
    1. Control_Negative: Control subjects with negative COVID self-report.
    2. Participant_Negative: Participant subjects with negative COVID self-report.
    3. Participant_Positive: Participant subjects with positive COVID self-report.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        Dict[str, Dict]: A dictionary mapping group labels to their respective data dictionaries.
    """
    groups_definition = {
        'Control_Negative': ('C', 'N'),  # Control subjects with negative COVID self-report
        'Participant_Negative': ('P', 'N'),  # Participant subjects with negative COVID self-report
        'Participant_Positive': ('P', 'Y')  # Participant subjects with positive COVID self-report
    }

    extracted_groups = {label: {} for label in groups_definition.keys()}

    for participant, content in data.items():
        participant_info = content.get('participant_info', {})
        control_participant_status = participant_info.get('Control/Participant')
        covid_self_report = participant_info.get('COVID self report')

        for label, (status, report) in groups_definition.items():
            if control_participant_status == status and covid_self_report == report:
                extracted_groups[label][participant] = content

    for label in groups_definition.keys():
        logging.info(f"{len(extracted_groups[label])} individuals in group '{label}'.")

    return extracted_groups


def reshape_cytokine_data(data: Dict) -> pd.DataFrame:
    """
    Reshapes cytokine data into a pandas DataFrame.
    """
    cytokine_rows = []
    for participant, content in data.items():
        cytokine_data = content.get('cytokine_data', {})
        participant_info = content.get('participant_info', {})
        for cytokine, details in cytokine_data.items():
            row = {
                'Participant_ID': participant,
                'Control_Participant': participant_info.get('Control/Participant'),
                'Cytokine_Marker': cytokine,
                'Value': details.get('Value'),
                'Missing': details.get('Missing', 0),
                'Outlier': details.get('Outlier', 0),
                'Extrapolated': details.get('Extrapolated', 0),
                'OOR_Status': details.get('OOR_Status', 0)
            }
            cytokine_rows.append(row)
    df = pd.DataFrame(cytokine_rows)
    if df.empty:
        logging.warning("Reshaped cytokine DataFrame is empty.")
    return df


def clean_cytokine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans cytokine data according to specified criteria:
    - Exclude markers where the value is null.
    - Exclude markers flagged as outliers.
    - Exclude markers with >80% OOR values across the dataset.
    - Exclude markers with >80% extrapolated values.
    """
    original_marker_count = df['Cytokine_Marker'].nunique()
    # Exclude rows where the value is null
    df = df[df['Value'].notnull()]

    # Exclude rows flagged as outliers
    df = df[df['Outlier'] != 1]

    # Exclude markers with >80% OOR values
    oor_marker_counts = df.groupby('Cytokine_Marker')['OOR_Status'].apply(
        lambda x: (x != 0).sum() / len(x)
    )
    markers_to_keep_oor = oor_marker_counts[oor_marker_counts <= 0.8].index
    df = df[df['Cytokine_Marker'].isin(markers_to_keep_oor)]

    # Exclude markers with >80% extrapolated values
    extrapolated_marker_counts = df.groupby('Cytokine_Marker')['Extrapolated'].apply(
        lambda x: x.sum() / len(x)
    )
    markers_to_keep_extrapolated = extrapolated_marker_counts[extrapolated_marker_counts <= 0.8].index
    df = df[df['Cytokine_Marker'].isin(markers_to_keep_extrapolated)]

    cleaned_marker_count = df['Cytokine_Marker'].nunique()
    logging.info(f"Cytokine markers before cleaning: {original_marker_count}")
    logging.info(f"Cytokine markers after cleaning: {cleaned_marker_count}")

    return df


def normalize_cytokine_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes cytokine values using z-score normalization.
    """
    if df.empty:
        logging.warning("Cytokine DataFrame is empty. Skipping normalization.")
        return df
    df['Z_Score'] = df.groupby('Cytokine_Marker')['Value'].transform(zscore)
    logging.info(f"Normalized cytokine data with {df['Z_Score'].count()} Z_Score values.")
    return df


def compare_cytokine_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, group1_label: str,
                            group2_label: str) -> pd.DataFrame:
    """
    Compares cytokine markers between two groups using statistical tests.

    Parameters:
        group1_df (pd.DataFrame): DataFrame for Group 1.
        group2_df (pd.DataFrame): DataFrame for Group 2.
        group1_label (str): Label for Group 1.
        group2_label (str): Label for Group 2.

    Returns:
        pd.DataFrame: DataFrame containing comparison results with adjusted p-values.
    """
    # Merge data for comparison
    combined_df = pd.concat([
        group1_df.assign(Group=group1_label),
        group2_df.assign(Group=group2_label)
    ])

    # Get the list of unique cytokine markers
    markers = combined_df['Cytokine_Marker'].unique()

    results = []

    for marker in markers:
        marker_data = combined_df[combined_df['Cytokine_Marker'] == marker]
        group1_values = marker_data[marker_data['Group'] == group1_label]['Z_Score'].dropna()
        group2_values = marker_data[marker_data['Group'] == group2_label]['Z_Score'].dropna()

        # Perform t-test if both groups have at least two samples
        if len(group1_values) > 1 and len(group2_values) > 1:
            t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
            result = {
                'Comparison': f'{group1_label} vs {group2_label}',
                'Cytokine_Marker': marker,
                f'{group1_label}_Mean_Z': group1_values.mean(),
                f'{group2_label}_Mean_Z': group2_values.mean(),
                'T_Statistic': t_stat,
                'P_Value': p_value
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values('P_Value', inplace=True)

    # Adjust p-values for multiple comparisons
    results_df = adjust_p_values(results_df)

    # Save the results
    results_df.to_csv('cytokine_comparison_results.csv', index=False)

    # Log the results
    logging.info("Cytokine Comparison Results:\n%s", results_df.head())

    return results_df


def reshape_autoantibody_data(data: Dict) -> pd.DataFrame:
    """
    Reshapes autoantibody data into a pandas DataFrame.
    """
    autoantibody_rows = []
    for participant, content in data.items():
        autoantibody_data = content.get('autoantibody_data', {})
        participant_info = content.get('participant_info', {})
        for ig_type, markers in autoantibody_data.items():
            for marker, value in markers.items():
                row = {
                    'Participant_ID': participant,
                    'Control_Participant': participant_info.get('Control/Participant'),
                    'Immunoglobulin_Type': ig_type,
                    'Autoantibody_Marker': marker,
                    'Value': value
                }
                autoantibody_rows.append(row)
    df = pd.DataFrame(autoantibody_rows)
    if df.empty:
        logging.warning("Reshaped autoantibody DataFrame is empty.")
    return df


def normalize_autoantibody_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes autoantibody values using z-score normalization for each immunoglobulin type and marker.
    """
    if df.empty:
        logging.warning("Autoantibody DataFrame is empty. Skipping normalization.")
        return df
    df['Z_Score'] = df.groupby(['Immunoglobulin_Type', 'Autoantibody_Marker'])['Value'].transform(zscore)
    logging.info(f"Normalized autoantibody data with {df['Z_Score'].count()} Z_Score values.")
    return df


def compare_autoantibody_groups(group1_df: pd.DataFrame, group2_df: pd.DataFrame, group1_label: str,
                                group2_label: str) -> pd.DataFrame:
    """
    Compares autoantibody markers between two groups using statistical tests, considering immunoglobulin types.

    Parameters:
        group1_df (pd.DataFrame): DataFrame for Group 1.
        group2_df (pd.DataFrame): DataFrame for Group 2.
        group1_label (str): Label for Group 1.
        group2_label (str): Label for Group 2.

    Returns:
        pd.DataFrame: DataFrame containing comparison results with adjusted p-values.
    """
    # Merge data for comparison
    combined_df = pd.concat([
        group1_df.assign(Group=group1_label),
        group2_df.assign(Group=group2_label)
    ])

    # Get the list of unique combinations of immunoglobulin types and markers
    combinations_unique = combined_df[['Immunoglobulin_Type', 'Autoantibody_Marker']].drop_duplicates()

    results = []

    for _, row in combinations_unique.iterrows():
        ig_type = row['Immunoglobulin_Type']
        marker = row['Autoantibody_Marker']
        marker_data = combined_df[
            (combined_df['Immunoglobulin_Type'] == ig_type) &
            (combined_df['Autoantibody_Marker'] == marker)
            ]
        group1_values = marker_data[marker_data['Group'] == group1_label]['Z_Score'].dropna()
        group2_values = marker_data[marker_data['Group'] == group2_label]['Z_Score'].dropna()

        # Perform t-test if both groups have at least two samples
        if len(group1_values) > 1 and len(group2_values) > 1:
            t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)
            result = {
                'Comparison': f'{group1_label} vs {group2_label}',
                'Immunoglobulin_Type': ig_type,
                'Autoantibody_Marker': marker,
                f'{group1_label}_Mean_Z': group1_values.mean(),
                f'{group2_label}_Mean_Z': group2_values.mean(),
                'T_Statistic': t_stat,
                'P_Value': p_value
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values('P_Value', inplace=True)

    # Adjust p-values for multiple comparisons
    results_df = adjust_p_values(results_df)

    # Save the results
    results_df.to_csv('autoantibody_comparison_results.csv', index=False)

    # Log the results
    logging.info("Autoantibody Comparison Results:\n%s", results_df.head())

    return results_df


def adjust_p_values(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts p-values for multiple comparisons using the Benjamini-Hochberg procedure.
    """
    if 'P_Value' not in results_df.columns:
        logging.error("P_Value column not found in results DataFrame.")
        return results_df

    p_values = results_df['P_Value']
    adjusted = multipletests(p_values, method='fdr_bh')
    results_df['Adjusted_P_Value'] = adjusted[1]
    results_df['Significant'] = adjusted[0]
    return results_df


def plot_significant_autoantibody_markers(results_df: pd.DataFrame, combined_df: pd.DataFrame, threshold=0.05):
    """
    Plots significant autoantibody markers between groups.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing comparison results with adjusted p-values.
        combined_df (pd.DataFrame): Combined DataFrame of all groups for plotting.
        threshold (float): Significance threshold for adjusted p-values.
    """
    significant = results_df[results_df['Adjusted_P_Value'] < threshold]
    for _, row in significant.iterrows():
        comparison = row['Comparison']
        ig_type = row['Immunoglobulin_Type']
        marker = row['Autoantibody_Marker']
        marker_data = combined_df[
            (combined_df['Immunoglobulin_Type'] == ig_type) &
            (combined_df['Autoantibody_Marker'] == marker)
            ]
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y='Z_Score', data=marker_data)
        plt.title(f'{ig_type} - {marker} ({comparison})')
        plt.xlabel('Group')
        plt.ylabel('Z-Score')
        plt.tight_layout()
        plt.savefig(f'{ig_type}_{marker}_comparison.png')
        plt.close()


def plot_significant_cytokine_markers(results_df: pd.DataFrame, combined_df: pd.DataFrame, threshold=0.05):
    """
    Plots significant cytokine markers between groups.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing comparison results with adjusted p-values.
        combined_df (pd.DataFrame): Combined DataFrame of all groups for plotting.
        threshold (float): Significance threshold for adjusted p-values.
    """
    significant = results_df[results_df['Adjusted_P_Value'] < threshold]
    for _, row in significant.iterrows():
        comparison = row['Comparison']
        marker = row['Cytokine_Marker']
        marker_data = combined_df[
            (combined_df['Cytokine_Marker'] == marker)
        ]
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y='Z_Score', data=marker_data)
        plt.title(f'{marker} ({comparison})')
        plt.xlabel('Group')
        plt.ylabel('Z-Score')
        plt.tight_layout()
        plt.savefig(f'{marker}_comparison.png')
        plt.close()


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
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    if group_labels is not None:
        # Ensure group_labels is a Series
        if isinstance(group_labels, pd.Index):
            group_labels = pd.Series(group_labels, name='Group')

        pca_df = pd.concat([pca_df, group_labels.reset_index(drop=True)], axis=1)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='PC1', y='PC2',
            hue='Group',
            palette='Set1',
            data=pca_df,
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


def plot_significant_markers(results_df: pd.DataFrame, combined_df: pd.DataFrame, threshold: float = 0.05,
                             marker_type: str = 'autoantibody'):
    """
    Plots significant markers between groups.

    Parameters:
        results_df (pd.DataFrame): DataFrame containing comparison results with adjusted p-values.
        combined_df (pd.DataFrame): Combined DataFrame of all groups for plotting.
        threshold (float): Significance threshold for adjusted p-values.
        marker_type (str): Type of marker ('autoantibody' or 'cytokine').
    """
    significant = results_df[results_df['Adjusted_P_Value'] < threshold]
    for _, row in significant.iterrows():
        comparison = row['Comparison']
        if marker_type == 'autoantibody':
            ig_type = row['Immunoglobulin_Type']
            marker = row['Autoantibody_Marker']
            marker_data = combined_df[
                (combined_df['Immunoglobulin_Type'] == ig_type) &
                (combined_df['Autoantibody_Marker'] == marker)
                ]
            title = f'{ig_type} - {marker} ({comparison})'
            filename = f'{ig_type}_{marker}_comparison.png'
        else:
            marker = row['Cytokine_Marker']
            marker_data = combined_df[
                (combined_df['Cytokine_Marker'] == marker)
            ]
            title = f'{marker} ({comparison})'
            filename = f'{marker}_comparison.png'

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y='Z_Score', data=marker_data,
                    palette='Set2' if marker_type == 'autoantibody' else 'Set3')
        plt.title(title)
        plt.xlabel('Group')
        plt.ylabel('Z-Score')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved plot for {title} as {filename}.")


def main(file_path: str):
    """
    Main function to process the data, perform comparisons, and execute PCA.

    Parameters:
        file_path (str): The path to the JSON data file.
    """
    # Set up logging in INFO mode
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load the JSON data from the provided file path
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        return
    except json.JSONDecodeError as e:
        logging.error("Error decoding JSON: %s", e)
        return

    # Apply filtering
    filtered_data = filter_individuals(data)

    # Extract the specific groups for comparison
    extracted_groups = extract_specific_groups(filtered_data)

    # Define the comparisons to perform
    comparisons = [
        ('Control_Negative', 'Participant_Negative'),
        ('Control_Negative', 'Participant_Positive')
    ]

    # --- Autoantibody Data Processing ---
    logging.info("Starting Autoantibody Data Processing...")

    autoantibody_results = []

    for group1_label, group2_label in comparisons:
        group1_df = normalize_autoantibody_data(reshape_autoantibody_data(extracted_groups[group1_label]))
        group2_df = normalize_autoantibody_data(reshape_autoantibody_data(extracted_groups[group2_label]))

        # Check if DataFrames are not empty
        if group1_df.empty or group2_df.empty:
            logging.warning(
                f"One of the autoantibody groups '{group1_label}' or '{group2_label}' is empty. Skipping comparison.")
            continue

        # Save the normalized data
        group1_df.to_csv(f'autoantibody_{group1_label}_zscore.csv', index=False)
        group2_df.to_csv(f'autoantibody_{group2_label}_zscore.csv', index=False)

        # Perform statistical comparison
        result_df = compare_autoantibody_groups(group1_df, group2_df, group1_label, group2_label)
        autoantibody_results.append(result_df)

    # Combine all comparison results
    if autoantibody_results:
        combined_auto_results = pd.concat(autoantibody_results, ignore_index=True)
        combined_auto_results.to_csv('autoantibody_comparison_results.csv', index=False)
        logging.info("All Autoantibody Comparisons Completed.")

        # Merge all autoantibody data for visualization
        combined_auto_df = pd.concat([
                                         group1_df.assign(Group=group1_label) for group1_label, group2_label in
                                         comparisons for group1_df in [
                normalize_autoantibody_data(reshape_autoantibody_data(extracted_groups[group1_label]))]
                                     ] + [
                                         group2_df.assign(Group=group2_label) for group1_label, group2_label in
                                         comparisons for group2_df in [
                normalize_autoantibody_data(reshape_autoantibody_data(extracted_groups[group2_label]))]
                                     ]).drop_duplicates()

        # Perform PCA for each immunoglobulin type
        for ig_type in combined_auto_df['Immunoglobulin_Type'].unique():
            ig_data = combined_auto_df[combined_auto_df['Immunoglobulin_Type'] == ig_type]
            pivot_df = ig_data.pivot(index='Participant_ID', columns='Autoantibody_Marker', values='Z_Score').dropna()

            # Extract group labels as a Series
            group_labels = pd.Series(
                pivot_df.index.map(lambda x:
                                   'Control_Negative' if x in extracted_groups['Control_Negative']
                                   else ('Participant_Negative' if x in extracted_groups['Participant_Negative']
                                         else 'Participant_Positive')),
                name='Group'
            )

            # Perform PCA
            pca, pca_df = perform_pca(
                df=pivot_df,
                n_components=2,
                group_labels=group_labels,
                title=f'PCA_Autoantibody_{ig_type}'
            )

            # Optionally, save the PCA-transformed data
            pca_df.to_csv(f'PCA_Autoantibody_{ig_type}.csv', index=False)
    else:
        logging.warning("No Autoantibody Comparisons were performed.")

    # --- Cytokine Data Processing ---
    logging.info("Starting Cytokine Data Processing...")

    cytokine_results = []

    for group1_label, group2_label in comparisons:
        group1_df = normalize_cytokine_data(clean_cytokine_data(reshape_cytokine_data(extracted_groups[group1_label])))
        group2_df = normalize_cytokine_data(clean_cytokine_data(reshape_cytokine_data(extracted_groups[group2_label])))

        # Check if DataFrames are not empty
        if group1_df.empty or group2_df.empty:
            logging.warning(
                f"One of the cytokine groups '{group1_label}' or '{group2_label}' is empty. Skipping comparison.")
            continue

        # Save the normalized data
        group1_df.to_csv(f'cytokine_{group1_label}_zscore.csv', index=False)
        group2_df.to_csv(f'cytokine_{group2_label}_zscore.csv', index=False)

        # Perform statistical comparison
        result_df = compare_cytokine_groups(group1_df, group2_df, group1_label, group2_label)
        cytokine_results.append(result_df)

    # Combine all comparison results
    if cytokine_results:
        combined_cytokine_results = pd.concat(cytokine_results, ignore_index=True)
        combined_cytokine_results.to_csv('cytokine_comparison_results.csv', index=False)
        logging.info("All Cytokine Comparisons Completed.")

        # Merge all cytokine data for visualization
        combined_cytokine_df = pd.concat([
                                             group1_df.assign(Group=group1_label) for group1_label, group2_label in
                                             comparisons for group1_df in [
                normalize_cytokine_data(clean_cytokine_data(reshape_cytokine_data(extracted_groups[group1_label])))]
                                         ] + [
                                             group2_df.assign(Group=group2_label) for group1_label, group2_label in
                                             comparisons for group2_df in [
                normalize_cytokine_data(clean_cytokine_data(reshape_cytokine_data(extracted_groups[group2_label])))]
                                         ]).drop_duplicates()

        total_markers = combined_cytokine_df['Cytokine_Marker'].nunique()
        logging.info(f"Total unique Cytokine Markers: {total_markers}")

        # Count the number of cytokine markers per Participant_ID
        marker_counts = combined_cytokine_df.groupby('Participant_ID')['Cytokine_Marker'].nunique()

        # Summary statistics
        logging.info(marker_counts)

        # Identify participants with incomplete data
        incomplete_participants = marker_counts[marker_counts < total_markers]
        logging.info(f"Number of participants with incomplete cytokine data: {incomplete_participants.count()}")

        # Optionally, list these participants
        logging.info("Participants with incomplete data:")
        logging.info(incomplete_participants.index.tolist())

        # Perform PCA on cytokine data
        pivot_cytokine_df = combined_cytokine_df.pivot(index='Participant_ID', columns='Cytokine_Marker',
                                                       values='Z_Score').dropna()

        # Extract group labels as a Series
        group_labels_cytokine = pd.Series(
            pivot_cytokine_df.index.map(lambda x:
                                        'Control_Negative' if x in extracted_groups['Control_Negative']
                                        else ('Participant_Negative' if x in extracted_groups['Participant_Negative']
                                              else 'Participant_Positive')),
            name='Group'
        )

        # Perform PCA
        pca_cytokine, pca_cytokine_df = perform_pca(
            df=pivot_cytokine_df,
            n_components=2,
            group_labels=group_labels_cytokine,
            title='PCA_Cytokine'
        )

        # Optionally, save the PCA-transformed data
        pca_cytokine_df.to_csv('PCA_Cytokine.csv', index=False)
    else:
        logging.warning("No Cytokine Comparisons were performed.")


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process immunology data.')
    parser.add_argument('file_path', type=str, help='Path to the JSON data file')
    args = parser.parse_args()

    # Run the main function
    main(args.file_path)
