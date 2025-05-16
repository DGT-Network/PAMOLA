"""
PAMOLA.CORE - Data Field Analysis Processor
---------------------------------------------------
This module provides an implementation of `BaseProfilingProcessor` for assessing privacy
characteristics of groups formed by quasi-identifiers,including pre-analysis for k-anonymity
and distribution of equivalence classes.
(C) 2024 Realm Inveo Inc. and DGT Network Inc.  

Licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause  
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Module: Named Entity Recognition Profiling Operations
--------------------------------   
It includes the following capabilities:  
- Group size distribution for each quasi-identifier set
- K-anonymity pre-assessment results
- Equivalence class distribution
- Records below k-threshold
- Diversity of sensitive attributes within groups
- Identification risk assessment
- Distribution of records by equivalence class size
- Visualization of k-distribution.

NOTE: Requires `pandas`.

Author: Realm Inveo Inc. & DGT Network Inc.
"""


from typing import Any, Dict, List, Optional
import pandas as pd

from scipy.stats import entropy
import logging

from pamola_core.profiling.base import BaseProfilingProcessor

# Configure logging
logger = logging.getLogger(__name__)


class PrivacyGroupAssessmentProfilingProcessor(BaseProfilingProcessor):
    """
    Processor for assessing privacy characteristics of groups formed by quasi-identifiers,
    including pre-analysis for k-anonymity and distribution of equivalence classes.
    """
    
    def __init__(
        self,
        quasi_identifier_sets: List[List[str]], 
        sensitive_attributes: Optional[List[str]] = None,
        calculate_k_distribution: Optional[bool] = True,
        min_k_threshold: Optional[int] = 2, 
        calculate_l_diversity: Optional[bool] = False, 
        identify_risk_groups: Optional[bool] = True,
        risk_threshold: Optional[float] = 0.5,
        save_equivalence_classes: Optional[bool] = False,
        max_classes_to_save: Optional[int] = 100,
    ):
        """
        Initializes the Privacy Group Assessment Profiling Processor

        Parameters:
        -----------
        quasi_identifier_sets : List[List[str]], required  
            Sets of fields considered quasi-identifiers (e.g., [["gender", "age_range", "zip"]])
        sensitive_attributes : List[str], optional  
            Fields considered sensitive (default=[]) (e.g., ["salary"], ["diagnosis", "income"])
        calculate_k_distribution : bool, optional  
            Whether to calculate k distribution (default=True)
        min_k_threshold : int, optional  
            Minimum k value threshold for reporting (default=2) (e.g., 3, 5)
        calculate_l_diversity : bool, optional  
            Whether to calculate l-diversity (default=False)
        identify_risk_groups : bool, optional  
            Whether to identify high-risk groups (default=True)
        risk_threshold : float, optional  
            Threshold for high-risk identification (default=0.5) (e.g., 0.3, 0.7)
        save_equivalence_classes : bool, optional  
            Whether to save equivalence class details (default=False)
        max_classes_to_save : int, optional  
            Maximum classes to save in detail (default=100) (e.g., 50, 200)

        """
        super().__init__()
        self.quasi_identifier_sets = quasi_identifier_sets
        self.sensitive_attributes = sensitive_attributes or []
        self.calculate_k_distribution = calculate_k_distribution
        self.min_k_threshold = min_k_threshold
        self.calculate_l_diversity = calculate_l_diversity
        self.identify_risk_groups = identify_risk_groups
        self.risk_threshold = risk_threshold
        self.save_equivalence_classes = save_equivalence_classes
        self.max_classes_to_save = max_classes_to_save
    
    def execute(self, df: pd.DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform privacy group analysis based on quasi-identifiers and sensitive attributes.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to analyze.
        **kwargs : dict
            Dynamic parameter overrides:
            
            - `quasi_identifier_sets` (List[List[str]], default=self.quasi_identifier_sets):
                Sets of fields considered quasi-identifiers.
            - `sensitive_attributes` (List[str], default=self.sensitive_attributes):
                Fields considered sensitive.
            - `calculate_k_distribution` (bool, default=self.calculate_k_distribution):
                Whether to calculate k distribution.
            - `min_k_threshold` (int, default=self.min_k_threshold):
                Minimum k value threshold for reporting.
            - `calculate_l_diversity` (bool, default=self.calculate_l_diversity):
                Whether to calculate l-diversity.
            - `identify_risk_groups` (bool, default=self.identify_risk_groups):
                Whether to identify high-risk groups.
            - `risk_threshold` (float, default=self.risk_threshold):
                Threshold for high-risk identification.
            - `save_equivalence_classes` (bool, default=self.save_equivalence_classes):
                Whether to save equivalence class details.
            - `max_classes_to_save` (int, default=self.max_classes_to_save):
                Maximum classes to save in detail.
        Returns:
        --------
        List[Dict[str, Any]]: A list of dictionaries, each containing profiling results
            such as k-distribution, l-diversity, and risk assessment for a given quasi-identifier set.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty. Returning empty results.")
            return []

        quasi_identifier_sets = kwargs.get("quasi_identifier_sets", self.quasi_identifier_sets)
        sensitive_attributes = kwargs.get("sensitive_attributes", self.sensitive_attributes)
        calculate_k_distribution = kwargs.get("calculate_k_distribution", self.calculate_k_distribution)
        min_k_threshold = kwargs.get("min_k_threshold", self.min_k_threshold)
        calculate_l_diversity = kwargs.get("calculate_l_diversity", self.calculate_l_diversity)
        identify_risk_groups = kwargs.get("identify_risk_groups", self.identify_risk_groups)
        risk_threshold = kwargs.get("risk_threshold", self.risk_threshold)
        save_equivalence_classes = kwargs.get("save_equivalence_classes", self.save_equivalence_classes)
        max_classes_to_save = kwargs.get("max_classes_to_save", self.max_classes_to_save)
        
        results = []
        for quasi_id_set in quasi_identifier_sets:
            
            # Validate that all columns in the quasi-identifier set exist in the DataFrame
            missing_columns = [col for col in quasi_id_set if col not in df.columns]
            if missing_columns:
                logger.error(f"Columns {missing_columns} are missing in the DataFrame.")
                raise KeyError(f"Columns {missing_columns} are missing in the DataFrame.")
            
            grouped = df.groupby(quasi_id_set, dropna=False)
            group_sizes = grouped.size()
            num_total_records = len(df)

            result: Dict[str, Any] = {
                "quasi_identifier_set": quasi_id_set
            }

            if calculate_k_distribution:
                result["k_distribution"] = self._calculate_k_distribution(
                        quasi_id_set, group_sizes, num_total_records, min_k_threshold
                    )
                

            if calculate_l_diversity and sensitive_attributes:
                result["l_diversity"] = self._calculate_l_diversity(grouped, sensitive_attributes)

            if identify_risk_groups:
                result["risk_assessment"] = self._calculate_risk_scores(group_sizes, risk_threshold, max_classes_to_save)

            if save_equivalence_classes and calculate_k_distribution:
                max_classes_to_save = min(max_classes_to_save, len(group_sizes))
                top_classes = group_sizes.nlargest(max_classes_to_save)
                result["sample_equivalence_classes"] = top_classes.reset_index().to_dict(orient="records")

            results.append(result)
        return results

    def _calculate_k_distribution(
        self,
        quasi_identifier_list: List[str],
        group_sizes: pd.Series,
        num_total_records: int,
        min_k_threshold: int
    ) -> Dict[str, Any]:
        """
        Calculate k-distribution and related metrics for privacy assessment.

        Parameters:
        -----------
        quasi_identifier_list : List[str]
            List of quasi-identifier columns used for grouping.
        group_sizes : pd.Series
            Series containing the size of each group (equivalence class).
        num_total_records : int
            Total number of records in the dataset.
        min_k_threshold : int
            Minimum k value threshold for compliance.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing k-distribution metrics, including:
            - Total equivalence classes
            - Group size distribution
            - Percent of records at risk
            - Compliance status
            - Number of groups below the k-threshold
        """
        # Calculate the distribution of equivalence class sizes
        equivalence_class_distribution = group_sizes.value_counts().sort_index()
        record_distribution_by_size = equivalence_class_distribution.to_dict()

        # Determine the minimum group size and compliance status
        min_k = group_sizes.min()
        compliant = min_k >= min_k_threshold

        # Calculate the number of records in small groups (below the threshold)
        records_in_small_groups = group_sizes[group_sizes < min_k_threshold].sum()

        # Return the k-distribution metrics
        return {
            "quasi_identifier_set": quasi_identifier_list,
            "total_equivalence_classes": len(group_sizes),
            "group_size_distribution": group_sizes.to_dict(),
            "k_distribution": equivalence_class_distribution.to_dict(),
            "distribution_by_class_size": record_distribution_by_size,
            "percent_records_at_risk": round((records_in_small_groups / num_total_records) * 100, 2)
            if num_total_records > 0 else 0,
            "min_k": min_k,
            "compliant": compliant,
            "records_in_small_groups": int(records_in_small_groups),
            "groups_below_threshold": int(group_sizes[group_sizes < min_k_threshold].count()),
        }
    
    def _calculate_l_diversity(
        self,
        grouped: Any,
        sensitive_attributes: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate l-diversity metrics for sensitive attributes within groups.

        Parameters:
        -----------
        grouped : pandas.pamola_core.groupby.DataFrameGroupBy
            Grouped DataFrame object created using `groupby` on quasi-identifiers.
        sensitive_attributes : List[str]
            List of sensitive attributes to analyze for diversity.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing l-diversity metrics for each sensitive attribute, including:
            - Unique counts of sensitive values per group
            - Entropy values (measuring diversity) per group
        """
        sensitive_diversity = {}

        # Iterate over each sensitive attribute to calculate diversity metrics
        for attr in sensitive_attributes:
            # Calculate normalized value counts for each group
            diversity_counts = grouped[attr].value_counts(normalize=True, observed=True).unstack(fill_value=0)

            # Calculate entropy (diversity measure) for each group
            entropy_values = diversity_counts.apply(lambda x: entropy(x[x > 0], base=2), axis=1)

            # Store unique counts and entropy values for the attribute
            sensitive_diversity[attr] = {
                "unique_counts": grouped[attr].nunique().to_dict(),
                "entropy": entropy_values.to_dict(),
            }

        # Return the l-diversity metrics
        return {"sensitive_attribute_diversity": sensitive_diversity}

    def _calculate_risk_scores(
        self,
        group_sizes: pd.Series,
        risk_threshold: float,
        max_classes_to_save: int
    ) -> Dict[str, Any]:
        """
        Calculate identification risk scores for each group and identify high-risk groups.

        Parameters:
        -----------
        group_sizes : pd.Series
            Series containing the size of each group (equivalence class).
        risk_threshold : float
            Threshold for identifying high-risk groups (e.g., 0.5).
        max_classes_to_save : int
            Maximum number of high-risk groups to save in the output.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing risk assessment metrics, including:
            - Risk scores for each group
            - High-risk groups exceeding the threshold
            - Number of high-risk groups
            - Top high-risk groups (sorted by risk score)
        """
        # Calculate risk scores for each group (1 / group size)
        risk_scores = group_sizes.map(lambda x: round(1.0 / x, 4) if x > 0 else float('inf'))

        # Identify high-risk groups where the risk score exceeds the threshold
        high_risk_groups = risk_scores[risk_scores >= risk_threshold]

        # Select the top N high-risk groups based on the risk score
        top_n_high_risk_groups = high_risk_groups.sort_values(ascending=False).head(max_classes_to_save)

        # Return the risk assessment metrics
        return {
            "identification_risk": {
                "risk_scores": risk_scores.to_dict(),
                "high_risk_groups": high_risk_groups.to_dict(),
                "num_high_risk_groups": len(high_risk_groups),
                "top_high_risk_groups": top_n_high_risk_groups.to_dict(),
            }
        }
