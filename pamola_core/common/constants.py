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
    OPERATION_NAMES = ["generalization", "noise_addition"]

    COMMON_DATE_FORMATS = [
        # ISO format with timezone
        "ISO8601",
        # 1. Date with full/short month names (English)
        "%d-%b-%Y",  # 01-Jan-2023
        "%d-%B-%Y",  # 01-January-2023
        "%d %b %Y",  # 01 Jan 2023
        "%d %B %Y",  # 01 January 2023
        "%B %d, %Y",  # January 1, 2023
        "%b %d, %Y",  # Jan 1, 2023
        "%d %B %Y %H:%M",  # 01 January 2023 16:30
        "%d %B %Y %I:%M %p",  # 01 January 2023 04:30 PM
        "%B %d, %Y %H:%M",  # January 1, 2023 16:30
        "%B %d, %Y %I:%M %p",  # January 1, 2023 04:30 PM
        # 2. DMY & MDY with 24-hour time
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m-%d-%Y %H:%M",
        "%m-%d-%Y %H:%M:%S",
        # 3. DMY & MDY no time
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        # 4. YMD formats
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        # 5. Non-separated
        "%Y%m%d",
        "%d%m%Y",
        # 6. With AM/PM
        "%d.%m.%Y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d %I:%M %p",
    ]

    FREQ_MAP = {"day": "D", "week": "W", "month": "MS", "quarter": "QS", "year": "YS"}
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
    Artifact_Category_Dictionary = "dictionary"
    Artifact_Category_Output = "output"
    Artifact_Category_Visualization = "visualization"
    Artifact_Category_Metrics = "metrics"
    Artifact_Category_Mapping = "mapping"
