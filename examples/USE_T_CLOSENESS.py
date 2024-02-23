# IN THIS MODULE, WE DEMONSTRATE HOW TO CALCULATE ANONYMIZATION METRICS: T-CLOSENESS
# IT USES THE DATA FOR THE MARKETING ATTRIBUTION TASK

# T-CLOSENESS AIMS TO ADDRESS THE LIMITATIONS OF L-DIVERSITY IN PREVENT-ING ATTRIBUTE DISCLOSURE. SPECIFICALLY,
# IT ENSURES THAT THE DISTRIBU-TION OF A SENSITIVE ATTRIBUTE WITHIN ANY EQUIVALENCE CLASS (GROUP OF INDISTINGUISHABLE
# RECORDS) IS CLOSE TO THE DISTRIBUTION OF THAT AT-TRIBUTE IN THE ENTIRE DATASET. THE DISTANCE BETWEEN THESE TWO
# DISTRIBUTIONS SHOULD NOT EXCEED A PREDEFINED THRESHOLD T. T-CLOSENESS USES THE EARTH MOVER DISTANCE MEASURE TO
# QUANTIFY THIS CLOSENESS

from metrics.anonymity_metrics import AnonymityMetrics
from utils.load_data import ma_bank

# [1] LOAD DATA FOR MARKETING ATTRIBUTION AND SHOW FIRST ROWS
print("----- > [1] LOAD DATASETS")

print(ma_bank.head())   # SHOW THAT WE HAVE DATA IS LOADED
# print(ma_bank_mask.head())
# print(ma_ads.head())
# print(ma_ads_mask.head())

initial_dataset_name = "MA_BANK"
quasi_identifiers_str = "SEX;AGE;INCOME;BRAND"  # Set the list of quasi-identifiers
sensitive_columns_str = "SHOPPINGDATE"          # Single sensitive column as a string


# [2] CALCULATE T-CLOSENESS FOR WHOLE DATASET
# /!\ The function takes a very long time to complete. To enable tracking, use the print_flag = true flag
print(f"----- > [2] CALCULATE T-CLOSENESS FOR DATASET {initial_dataset_name}")
# Calculate the metric with AnonymityMetrics class
t_closeness_value = AnonymityMetrics.calculate_t_closeness(
    ma_bank, quasi_identifiers_str, sensitive_columns_str, 3,print_flag=True
)

print(f"Calculated t-Closeness: {t_closeness_value}")
