# THIS IS MODULE TO CALCULATE ENTROPY WITH DIFFERENT WAY
# INFORMATION ENTROPY IS A MEASURE OF THE UNCERTAINTY OR UNEXPECTEDNESS OF IN-FORMATION. IT DETERMINES HOW MUCH
# INFORMATION IS CONTAINED IN A MESSAGE OR DATA SOURCE. THE HIGHER THE ENTROPY, THE MORE INFORMATION THERE IS AND
# THE MORE DIFFICULT IT IS TO PREDICT OR COMPRESS. INFORMATION ENTROPY IS MEASURED IN BITS AND CAN BE CALCULATED USING
# SHANNON'S FORMULA: H(X) = - SUM(p(x)*log_2(p(x))).  Where x <-X, where X is the  set of possible values and p(x) is
# the probability that the value of x will occur.

from utils.load_data import ma_bank
from utils.entropy import calculate_entropy
from utils.entropy import calculate_renyi_entropy
from utils.entropy import calculate_conditional_entropy

initial_dataset_name = "MA_BANK"
field_list_str = "SEX;AGE;INCOME;BRAND"  # Set the list of attributes
conditional_field_list_str = "SHOPPINGDATE;PURCHASEAMOUNT"

# [1] LOAD DATA FOR MARKETING ATTRIBUTION AND SHOW FIRST ROWS
print("----- > [1] LOAD DATASETS")

print(ma_bank.head())

# [2] CALCULATE SHANNON ENTROPY
# The Shannon entropy can be used to assess the quality of data anonymization. Data anony-mization is the process by
# which personal or sensitive information about individuals is deleted or modified to make them unidentifiable.
# Information entropy can show how well data is anonymized by measuring the diversity or heterogeneity of the data.
# The higher the entropy, the greater the anonymity and the lower the risk of identification.
print(f"----- > [2] CALCULATE SHANNON ENTROPY FOR DATASET {initial_dataset_name} AND FIELD LIST {field_list_str}")

shannon_entropy = calculate_entropy(ma_bank, field_list_str)
print(f"Calculated Shanon Entropy: {shannon_entropy}")

# [3] CALCULATE RÉNYI ENTROPY FOR THE SAME DATASET AND FIELD LIST
# Renyi's entropy generalizes the concept of Shannon's entropy and introduces the α parameter, which allows the
# calculation to be adapted to different tasks and conditions. The higher-order Renyi entropy (large values of alfa) is
# less sensitive to unlikely events compared to the Shannon entropy.
print(f"----- > [3] CALCULATE RÉNYI ENTROPY FOR DATASET {initial_dataset_name} AND FIELD LIST {field_list_str}")
renyi_entropy = calculate_renyi_entropy(ma_bank, field_list_str, alpha=2)  # Alpha is set to 2 for collision entropy
print(f"Calculated Rényi Entropy (alpha=2): {renyi_entropy}")

# [4] CALCULATE CONDITIONAL ENTROPY FOR THE SAME DATASET AND FIELD LIST
# Conditional entropy is a measure of how much uncertainty remains about a random variable Y after observing another
# random variable X. It can be calculated as the average of the entropy of Y for each possible value of X.
# Mathematically, it is written as: H(Y|X) = SUM_x (p(x) H(Y|X=x)), where p(x) is the probability of X taking the
# value x, and H(Y∣X=x) is the entropy of Y given that X equals x. Conditional entropy is always less than or equal
# to the entropy of Y, and is zero if and only if Y is completely determined by X. Conditional entropy can be used
# to quantify the amount of information gained or lost by conditioning on a variable.

# If all the values of the sensitive attribute in the group are the same for each unique set of quasi-identifier
# values, then the conditional entropy will be 0. This means that the sensitive attribute is completely dependent on
# quasi-identifiers. For example, if for each unique combination of quasi-identifiers "SEX; AGE; INCOME; BRAND" there
# is only one unique record "SHOPPINGDATE; PUR-CHASEAMOUNT", the conditional entropy will also be 0. This indicates a
# lack of diversity in sensitive data with known quasi-identifiers.

print(f"----- > [4] CALCULATE CONDITIONAL ENTROPY FOR DATASET {initial_dataset_name} AND FIELD LIST {field_list_str}")
conditional_entropy = calculate_conditional_entropy(ma_bank, field_list_str,
                                                    conditional_field_list_str, True)
print(f"Calculated Conditional Entropy: {conditional_entropy}")
