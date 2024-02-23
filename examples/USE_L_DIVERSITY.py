# IN THIS MODULE, WE DEMONSTRATE HOW TO CALCULATE ANONYMIZATION METRICS: L-DIVERSITY
# IT USES THE DATA FOR THE MARKETING ATTRIBUTION TASK

# CALCULATION OF L-DIVERSITY. L-DIVERSITY IS A PROPERTY OF A DATASET THAT MEASURES THE DIVERSITY OF SENSITIVE
# VALUES FOR EACH COLUMN IN WHICH THEY OCCUR. A DATASET HAS L-DIVERSITY IF, FOR EACH COLLECTION OF ROWS WITH THE SAME
# QUASI-IDENTIFIERS, THERE ARE AT LEAST L-DIFFERENT VALUES FOR COMBINATIONS OF SENSITIVE ATTRIBUTES (OR ATTRIBUTES).
# THIS UNDERSTANDING CORRESPONDS TO AN UNDERSTANDING CALLED DISTINCT L-DIVERSITY

from metrics.anonymity_metrics import AnonymityMetrics
from utils.load_data import ma_bank

# [1] LOAD DATA FOR MARKETING ATTRIBUTION AND SHOW FIRST ROWS
print("----- > [1] LOAD DATASETS")

print(ma_bank.head())
# print(ma_bank_mask.head())
# print(ma_ads.head())
# print(ma_ads_mask.head())

initial_dataset_name = "MA_BANK"
quasi_identifiers_str = "SEX;AGE;INCOME;BRAND"  # Set the list of quasi-identifiers
sensitive_columns_str = "SHOPPINGDATE;PURCHASEAMOUNT"  # Set the group of sensitive attributes

# [2] CALCULATE L-DIVERSITY FOR EVERY ROW OF DATASET AND ADD NEW ATTRIBUTE WITH L-DIVERSITY VALUE
print(f"----- > [2] CALCULATE L-DIVERSITY FOR EACH ROW FOR DATASET {initial_dataset_name}")

df, count_below, percent_below = AnonymityMetrics.calculate_l_diversity(
    ma_bank, quasi_identifiers_str, sensitive_columns_str, threshold=2, print_flag=True
)

print(f"Count below threshold: {count_below}, Percent below threshold: {percent_below}%")
# Inspect the result
print(df.head())  # Print the result to see its structure

# [3] CALCULATE GLOBAL VALUE FOR L-DIVERSITY (FOR WHOLE DATASET)
print(f"----- > [3] CALCULATE L-DIVERSITY FOR WHOLE DATASET {initial_dataset_name} (i.e. GLOBALLY)")
# Calculate global L-Diversity using 'min' statistic
global_l_diversity_min = AnonymityMetrics.calculate_global_l_diversity(
    ma_bank, quasi_identifiers_str, sensitive_columns_str, statistic='min', print_flag=True
)

# Calculate global L-Diversity using 'mean' statistic
global_l_diversity_mean = AnonymityMetrics.calculate_global_l_diversity(
    ma_bank, quasi_identifiers_str, sensitive_columns_str, statistic='mean', print_flag=True
)

# Calculate global L-Diversity using 'median' statistic
global_l_diversity_median = AnonymityMetrics.calculate_global_l_diversity(
    ma_bank, quasi_identifiers_str, sensitive_columns_str, statistic='median', print_flag=True
)

print(f"Global L-Diversity (Min): {global_l_diversity_min}")
print(f"Global L-Diversity (Mean): {global_l_diversity_mean}")
print(f"Global L-Diversity (Median): {global_l_diversity_median}")
