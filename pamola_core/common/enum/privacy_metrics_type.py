from enum import Enum

class PrivacyMetricsType(str, Enum):
    """
    Enum representing supported privacy metrics for evaluating
    the privacy preservation of data post-transformation.

    Distance-based Privacy Metrics:
    - DCR: Distance to Closest Record
    - NNDR: Nearest Neighbor Distance Ratio

    Uniqueness-based Privacy Metrics:
    - IDENTITY_DISCLOSURE: Risk of re-identification
    - K_ANONYMITY: Each record is indistinguishable from at least k-1 others
    - L_DIVERSITY: Sensitive attributes within each equivalence class are diverse
    - T_CLOSENESS: Distribution of sensitive attributes is close to the overall
    """

    # Distance-based Privacy Metrics
    DCR = "dcr"                            # Distance to Closest Record
    NNDR = "nndr"                          # Nearest Neighbor Distance Ratio

    # Uniqueness-based Privacy Metrics
    UNIQUENESS = "uniqueness"  # Risk of re-identification
    K_ANONYMITY = "k_anonymity"                  # k-Anonymity
    L_DIVERSITY = "l_diversity"                  # l-Diversity
    T_CLOSENESS = "t_closeness"                  # t-Closeness
