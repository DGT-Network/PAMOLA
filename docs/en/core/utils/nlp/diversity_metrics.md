from pamola_core.utils.nlp.diversity_metrics import (
    calculate_semantic_diversity,
    calculate_lexical_diversity,
    calculate_token_diversity
)
import pandas as pd

def assess_text_anonymization_quality(original_df, anonymized_df, text_column):
    """
    Assess the quality of text anonymization using diversity metrics.
    """
    results = {}
    
    # 1. Semantic diversity change
    orig_semantic = calculate_semantic_diversity(
        original_df[text_column],
        method="token_overlap"
    )
    anon_semantic = calculate_semantic_diversity(
        anonymized_df[text_column],
        method="token_overlap"
    )
    results['semantic_diversity_loss'] = orig_semantic - anon_semantic
    
    # 2. Lexical diversity change
    orig_lexical = calculate_lexical_diversity(
        original_df[text_column].tolist(),
        method="mtld"
    )
    anon_lexical = calculate_lexical_diversity(
        anonymized_df[text_column].tolist(),
        method="mtld"
    )
    results['lexical_diversity_loss'] = orig_lexical - anon_lexical
    
    # 3. Token distribution change
    orig_token_div = calculate_token_diversity(
        original_df[text_column],
        method="all"
    )
    anon_token_div = calculate_token_diversity(
        anonymized_df[text_column],
        method="all"
    )
    
    results['entropy_change'] = (
        anon_token_div['entropy'] - orig_token_div['entropy']
    )
    results['simpson_change'] = (
        anon_token_div['simpson'] - orig_token_div['simpson']
    )
    
    # 4. Quality assessment
    if results['semantic_diversity_loss'] > 0.5:
        results['quality_assessment'] = "High information loss"
    elif results['semantic_diversity_loss'] > 0.3:
        results['quality_assessment'] = "Moderate information loss"
    else:
        results['quality_assessment'] = "Acceptable information loss"
    
    return results

# Example usage
original_data = pd.DataFrame({
    'description': [
        "Software engineer with Python experience",
        "Data scientist specializing in machine learning",
        "DevOps engineer with AWS certification",
        "Frontend developer skilled in React"
    ]
})

anonymized_data = pd.DataFrame({
    'description': [
        "Technical role with programming experience",
        "Analytics role with advanced skills",
        "Infrastructure role with cloud certification",
        "Development role with framework skills"
    ]
})

quality_metrics = assess_text_anonymization_quality(
    original_data,
    anonymized_data,
    'description'
)

print("Anonymization Quality Assessment:")
for metric, value in quality_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.3f}")
    else:
        print(f"  {metric}: {value}")