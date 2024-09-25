import pandas as pd
import json
import sys


def main(file_path):
    # Load the JSON data from the provided file path
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Lists to store cytokine and autoantibody data for comparison
    cytokine_rows = []
    autoantibody_rows = []

    # Loop through each participant in the data
    for participant, details in data.items():
        participant_info = details['participant_info']
        cytokine_data = details.get('cytokine_data', {})
        autoantibody_data = details.get('autoantibody_data', {})

        # Determine if the individual is a control or participant
        control_participant_label = participant_info.get('Control/Participant', 'Unknown')

        # Add cytokine data
        cytokine_row = {
            'Participant': participant,
            'Control/Participant': control_participant_label,
            'Sex': participant_info.get('Sex'),
            'Age': participant_info.get('Age at time of consent'),
        }

        # Add each cytokine as a separate column
        for cytokine, values in cytokine_data.items():
            cytokine_row[cytokine] = values.get('Value')
        cytokine_rows.append(cytokine_row)

        # Add autoantibody data
        for ab_type, antibodies in autoantibody_data.items():
            autoantibody_row = {
                'Participant': participant,
                'Control/Participant': control_participant_label,
                'Sex': participant_info.get('Sex'),
                'Age': participant_info.get('Age at time of consent'),
                'Antibody Type': ab_type
            }
            # Add each antibody as a separate column
            for antibody, value in antibodies.items():
                autoantibody_row[antibody] = value
            autoantibody_rows.append(autoantibody_row)

    # Create DataFrames from the lists of rows
    df_cytokines = pd.DataFrame(cytokine_rows)
    df_autoantibodies = pd.DataFrame(autoantibody_rows)

    # Display first few rows of each DataFrame
    print("Cytokines DataFrame (wide format):\n", df_cytokines.head())
    print("Autoantibodies DataFrame (wide format):\n", df_autoantibodies.head())

    # Save each DataFrame to a CSV file for later use
    df_cytokines.to_csv('cytokines_data_comparison.csv', index=False)
    df_autoantibodies.to_csv('autoantibodies_data_comparison.csv', index=False)


if __name__ == '__main__':
    # Check if the file path is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)

    # Get the file path from the command-line arguments
    file_path = sys.argv[1]

    # Run the main function
    main(file_path)
