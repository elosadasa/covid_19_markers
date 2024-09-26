import pandas as pd
import json
import argparse
import logging
import matplotlib.pyplot as plt
from typing import Dict, Tuple


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


def stratify_individuals(data: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Stratifies individuals into control, no COVID, and COVID groups.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        Tuple[Dict, Dict, Dict]: Dictionaries for control, no COVID, and COVID groups.
    """
    control_group = {}
    no_covid_group = {}
    covid_group = {}

    for participant, content in data.items():
        participant_info = content.get('participant_info', {})
        control_participant_status = participant_info.get('Control/Participant')
        covid_self_report = participant_info.get('COVID self report')

        if control_participant_status == 'C':
            control_group[participant] = content
        elif control_participant_status == 'P':
            if covid_self_report == 'N':
                no_covid_group[participant] = content
            elif covid_self_report == 'Y':
                covid_group[participant] = content

    # Log the count of individuals in each group
    logging.info(f'{len(control_group)} individuals in the control group.')
    logging.info(f'{len(no_covid_group)} individuals in the no COVID group.')
    logging.info(f'{len(covid_group)} individuals in the COVID group.')

    return control_group, no_covid_group, covid_group


def plot_covid_self_report_distribution(data: Dict) -> pd.Series:
    """
    Calculates and plots the distribution of participants according to the COVID self-report.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        pd.Series: The distribution of COVID self-reports.
    """
    covid_reports = [
        content.get('participant_info', {}).get('COVID self report')
        for content in data.values()
    ]
    covid_report_series = pd.Series(covid_reports).dropna()

    # Calculate the distribution
    distribution = covid_report_series.value_counts()

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    distribution.plot(kind='bar', color='skyblue')
    plt.title('COVID Self-Report Distribution of Participants')
    plt.xlabel('COVID Self-Report')
    plt.ylabel('Number of Participants')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('covid_self_report_distribution.png')
    plt.close()

    # Log the distribution
    logging.info('COVID self-report distribution:\n%s', distribution.to_string())
    return distribution


def reshape_cytokine_data(data: Dict) -> pd.DataFrame:
    """
    Reshapes cytokine data into a pandas DataFrame.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        pd.DataFrame: The reshaped cytokine data.
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
                'Missing': details.get('Missing'),
                'Outlier': details.get('Outlier'),
                'Extrapolated': details.get('Extrapolated'),
                'OOR_Status': details.get('OOR_Status')
            }
            cytokine_rows.append(row)
    return pd.DataFrame(cytokine_rows)


def reshape_autoantibody_data(data: Dict) -> pd.DataFrame:
    """
    Reshapes autoantibody data into a pandas DataFrame.

    Parameters:
        data (dict): The input data dictionary.

    Returns:
        pd.DataFrame: The reshaped autoantibody data.
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
    return pd.DataFrame(autoantibody_rows)


def main(file_path: str):
    """
    Main function to process the data.

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

    # Plot the COVID self-report distribution for the filtered individuals
    plot_covid_self_report_distribution(filtered_data)

    # Apply stratification
    control_group, no_covid_group, covid_group = stratify_individuals(filtered_data)

    # Reshape the cytokine data for each group
    cytokine_control_df = reshape_cytokine_data(control_group)
    cytokine_no_covid_df = reshape_cytokine_data(no_covid_group)
    cytokine_covid_df = reshape_cytokine_data(covid_group)

    # Reshape the autoantibody data for each group
    autoantibody_control_df = reshape_autoantibody_data(control_group)
    autoantibody_no_covid_df = reshape_autoantibody_data(no_covid_group)
    autoantibody_covid_df = reshape_autoantibody_data(covid_group)

    # Display the first few rows of each DataFrame (for one group as example)
    logging.info("Cytokine Data for Control Group:\n%s", cytokine_control_df.head())
    logging.info("Autoantibody Data for Control Group:\n%s", autoantibody_control_df.head())


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process immunology data.')
    parser.add_argument('file_path', type=str, help='Path to the JSON data file')
    args = parser.parse_args()

    # Run the main function
    main(args.file_path)
