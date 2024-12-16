# Cytokine and Autoantibody Data Processing

This script processes cytokine and autoantibody data to identify significant markers, perform statistical comparisons, and generate visualizations like PCA plots. It supports data cleaning, imputation, statistical testing, and dimensionality reduction for immunology datasets.

---

## Features

1. **Data Loading and Cleaning**:
   - Loads cytokine and autoantibody data from CSV files.
   - Filters rows with missing key information (e.g., patient demographics).
   - Removes markers with a high proportion of out-of-range (OOR) values.

2. **Imputation**:
   - Handles missing data using median or KNN-based imputation based on the proportion of missingness.

3. **Normalization**:
   - Normalizes data using Z-score for consistent scaling.

4. **Group Analysis**:
   - Segments data into groups based on patient characteristics (e.g., `Control_Negative` and `Participant_Positive`).
   - Reshapes data from wide to long format for group comparisons.

5. **Statistical Testing**:
   - Performs statistical tests (e.g., t-tests, Mann-Whitney U) to identify significant differences between groups.
   - Adjusts p-values for multiple comparisons using the Benjamini-Hochberg method.

6. **PCA Visualization**:
   - Performs Principal Component Analysis (PCA) to visualize group differences.
   - Saves PCA plots and transformed datasets for further analysis.

7. **Results Export**:
   - Exports statistical comparison results and PCA data to CSV files.
   - Generates PCA plots for significant markers.

---

## Requirements

- Python 3.8+
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - scikit-learn
  - argparse

Install dependencies via `pip`:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn argparse
```
## Usage

1. **Prepare Input Files**:
   - Cytokine data in CSV format.
   - Autoantibody data in CSV format.

2. **Run the Script**:
   Execute the script from the command line:
```bash
python script_name.py <cytokine_file> <autoantibody_file>
```
Replace <cytokine_file> and <autoantibody_file> with the paths to the respective CSV files.
3. **Outputs**:
   - **Statistical Comparison Results**:
     - `cytokine_comparison_results.csv`
     - `autoantibody_comparison_results.csv`

   - **PCA Data**:
     - `PCA_Cytokine.csv`
     - `PCA_Autoantibody.csv`

   - **PCA Plots**:
     - `PCA_Cytokine.png`
     - `PCA_Autoantibody.png`
     - `PCA_Cytokine_Significant_Markers.png`
     - `PCA_Autoantibody_Significant_Markers.png`