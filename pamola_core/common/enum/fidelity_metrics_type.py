from enum import Enum

class FidelityMetricsType(str, Enum):
    """
    Enum representing supported fidelity metrics type for comparing
    statistical similarity between original and anonymized datasets.
    
    Members:
    - KS: Kolmogorov-Smirnov test
    - KL: Kullback-Leibler divergence
    - JS: Jensen-Shannon divergence (optional/expandable)
    - WASSERSTEIN: Wasserstein distance (optional/expandable)
    """

    KS = "ks"                # Kolmogorov-Smirnov Test
    KL = "kl"                # Kullback-Leibler Divergence
    JS = "js"                # Jensen-Shannon Divergence (optional)
    WASSERSTEIN = "wasserstein"  # Wasserstein Distance (optional)