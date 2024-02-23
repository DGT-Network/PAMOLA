# IN THIS MODULE, WE DEMONSTRATE HOW TO CALCULATE ANONYMIZATION METRICS: K-ANONYMITY
# IT USES THE DATA FOR THE MARKETING ATTRIBUTION TASK

# K-ANONYMITY ENSURES THAT WHEN RELEASING DATA, EACH PERSON’S INFORMATION CANNOT BE DISTINGUISHED FROM AT LEAST
# K OTHER INDIVIDUALS WHOSE INFORMATION ALSO APPEARS IN THE RELEASE.

from metrics.anonymity_metrics import AnonymityMetrics
from utils.load_data import ma_bank, ma_bank_mask, ma_ads, ma_ads_mask

# [1] LOAD DATA FOR MARKETING ATTRIBUTION AND SHOW FIRST ROWS
print("----- > [1] LOAD DATASETS")

print(ma_bank.head())
print(ma_bank_mask.head())
print(ma_ads.head())
print(ma_ads_mask.head())

initial_dataset_name = "MA_BANK"
quasi_identifiers_str = "SEX;AGE;INCOME;BRAND"  # Set the list of quasi-identifiers
sensitive_columns_str = "SHOPPINGDATE;PURCHASEAMOUNT"  # Set the group of sensitive attributes

# [2] CALCULATE K-ANONYMITY METRICS FOR EACH ROW (AS THE SIZE OF THE EQUIVALENCE CLASS)
# AND PLACE AN ADDITIONAL COLUMN IN THE SOURCE DATAFRAME
print(f"----- > [2] CALCULATE K-ANONYMITY FOR EACH ROW OF DATASET {initial_dataset_name}")

threshold = 3                                   # Minimum for k-anonymity

anonymity_metrics = AnonymityMetrics()          # Create the class instance of Anonymity Metric
ma_bank, k_min, count_below_threshold, percent_below_threshold = \
    anonymity_metrics.calculate_k_anonymity(ma_bank, quasi_identifiers_str,
                                            threshold=threshold, print_flag=True)
print(ma_bank.head())

# [3] CALCULATE THE K-ANONYMITY FOR THE ENTIRE SET, GET THE MINIMUM VALUE (PROBABILITY OF RE-IDENTIFICATION),
# MEAN (UNIFORMITY SCORE), MEDIAN VALUE (HOW TYPICAL THE DATA IS)
print(f"----- > [3] CALCULATE GLOBAL K-ANONYMITY FOR WHOLE DATASET {initial_dataset_name}")

# Calculation minimal k-anonymity (re-identification probability P_re-id = 1/(k_min) )
k_min = anonymity_metrics.calculate_global_k_anonymity(ma_bank, quasi_identifiers_str, stat_type='min')
print(f"Minimum k-anonymity value: {k_min}")

# Calculation mean k-anonymity (uniformity score)
k_mean = anonymity_metrics.calculate_global_k_anonymity(ma_bank, quasi_identifiers_str, stat_type='mean')
print(f"Mean k-anonymity value: {k_mean}")

# Calculation median for k (diversity score)
k_median = anonymity_metrics.calculate_global_k_anonymity(ma_bank, quasi_identifiers_str, stat_type='median')
print(f"Median k-anonymity value: {k_median}")