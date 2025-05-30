"""
PAMOLA.CORE - Constants Module
---------------------------------------------------------
This module defines global constants used throughout the project to improve 
maintainability and reduce hardcoded values in the codebase.

Features:
 - Centralized operation names to ensure consistency across modules.
 - Prevent hardcoded strings and facilitate easy updates.

This module is useful for logging, data transformations, and privacy-preserving 
operations where standardized operation names are required.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

from typing import List


class Constants:
    OPERATION_NAMES = [
        "generalization",
        "noise_addition"
    ]
    COMMON_DATE_FORMATS: List[str] = [
        "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"
    ]
    FREQ_MAP = {
        "day": "D",
        "week": "W",
        "month": "MS",
        "quarter": "QS",
        "year": "YS"
    }
    LOG_DIR = "logs"
    DISTRIBUTION_LABELS = {
        "distribution_similarity_score": "Distribution Similarity Score",
        "kl_divergence_orig_mid": "KL Divergence (Original vs Midpoint)",
        "kl_divergence_gen_mid": "KL Divergence (Generated vs Midpoint)",
        "uniqueness_preservation": "Uniqueness Preservation",
        "entropy_original": "Entropy (Original)",
        "entropy_generated": "Entropy (Generated)",
        "gini_original": "Gini Coefficient (Original)",
        "gini_generated": "Gini Coefficient (Generated)",
        "top_value_overlap@10": "Top-10 Value Overlap",
    }

    # Artifact Category
    Artifact_Category_Dictionary = 'dictionary'
    Artifact_Category_Output = 'output'
    Artifact_Category_Visualization = 'visualization'
    Artifact_Category_Metrics = 'metrics'
    Artifact_Category_Mapping = 'mapping'

