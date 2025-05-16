#!/usr/bin/env python
"""
PAMOLA (Pattern Anonymization and Profiling for Large-scale Applications) - Pattern Profiling Example
----------------------------------------------------------
This script demonstrates how to use the PatternProfiler for
analyzing patterns in categorical fields, particularly for
resume and job posting data.

Examples include:
1. Basic pattern analysis on a single-value field
2. Multi-value field analysis with validation
3. Custom pattern detection and visualization
4. Advanced usage with Dask for large files

(C) 2025 BDA

Author: V.Khvatov
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import pamola modules
from pamola_core.profiling.categorical.pattern import PatternProfiler
from pamola_core.config import config
from pamola_core.utils.logging import configure_logging


def example_basic_pattern_analysis():
    """
    Example 1: Basic pattern analysis on a single-value field.
    """
    print("\n=== Example 1: Basic Pattern Analysis ===")

    # Create sample data
    data = pd.DataFrame({
        'gender': ['Male', 'Female', 'male', 'FEMALE', 'Male', 'Non-binary',
                   'MALE', 'female', 'Male', 'Unknown', 'Prefer not to say',
                   None, 'female', 'Male', 'Female']
    })

    # Create profiler with allowed values
    profiler = PatternProfiler(
        field_name="gender",
        input_file=data,
        allowed_values="Male;Female;Non-binary",
        case_sensitive=False,
        allow_nulls=True
    )

    # Run profiling
    results = profiler.profile()

    # Print key results
    print(f"Total values: {results['total_values']}")
    print(f"Null values: {results['null_count']}")
    print(f"Matching values: {results['matching_count']} ({results['matching_percentage']:.1f}%)")
    print(f"Non-matching values: {results['non_matching_count']}")
    print("\nMost common patterns:")
    for pattern, count in results['most_common_patterns'].items():
        print(f"  {pattern}: {count}")

    # Create and save visualization
    output_path = profiler.visualize_results(
        output_name="gender_pattern_analysis"
    )
    print(f"\nVisualization saved to: {output_path}")


def example_multi_value_field_analysis():
    """
    Example 2: Multi-value field analysis with validation.
    """
    print("\n=== Example 2: Multi-Value Field Analysis ===")

    # Create sample data
    data = pd.DataFrame({
        'skills': [
            'Python, SQL, Machine Learning',
            'Java, JavaScript, HTML, CSS',
            'C++, Python, Data Structures',
            'Excel, Word, PowerPoint',
            'Python, R, Statistics, Data Analysis',
            'Project Management, Leadership, Communication',
            None,
            'Machine Learning, AI, Deep Learning, Python',
            'SQL, Database Design, MongoDB, PostgreSQL',
            'Unknown, Test Skill'
        ]
    })

    # Define allowed values
    allowed_skills = [
        'Python', 'SQL', 'Machine Learning', 'Java', 'JavaScript',
        'HTML', 'CSS', 'C++', 'Data Structures', 'Excel', 'Word',
        'PowerPoint', 'R', 'Statistics', 'Data Analysis',
        'Project Management', 'Leadership', 'Communication',
        'AI', 'Deep Learning', 'Database Design', 'MongoDB', 'PostgreSQL'
    ]

    # Create profiler
    profiler = PatternProfiler(
        field_name="skills",
        input_file=data,
        allowed_values=allowed_skills,
        multi_value_separator=",",
        case_sensitive=False,
        save_non_matching=True,
        non_matching_output_name="invalid_skills"
    )

    # Run profiling
    results = profiler.profile()

    # Print key results
    print(f"Total records: {results['total_values']}")
    print(f"Records with invalid skills: {results['non_matching_count']}")
    print(f"Total distinct skills: {len(results.get('most_common_subvalues', {}))}")
    print(f"Average skills per record: {results.get('avg_values_per_field', 0):.2f}")

    print("\nMost common skills:")
    for skill, count in list(results.get('most_common_subvalues', {}).items())[:5]:
        print(f"  {skill}: {count}")

    # Check for non-matching file
    if 'non_matching_file' in results:
        print(f"\nNon-matching records saved to: {results['non_matching_file']}")

    # Create and save visualization
    output_path = profiler.visualize_results(
        output_name="skills_pattern_analysis"
    )
    print(f"\nVisualization saved to: {output_path}")


def example_custom_pattern_detection():
    """
    Example 3: Custom pattern detection.
    """
    print("\n=== Example 3: Custom Pattern Detection ===")

    # Create sample data
    data = pd.DataFrame({
        'employee_id': [
            'EMP-123456',
            'TEMP-1234',
            'EMP-987654',
            'ABC12345',
            'TEMP-5678',
            'EMP-456789',
            '12345678',
            'CONTRACTOR-123',
            'TEMP-9876',
            'EMP-234567'
        ]
    })

    # Define custom patterns
    custom_patterns = [
        {"name": "emp_id_format", "regex": r"^EMP-\d{6}$"},
        {"name": "temp_id_format", "regex": r"^TEMP-\d{4}$"},
        {"name": "contractor_id_format", "regex": r"^CONTRACTOR-\d{3}$"}
    ]

    # Create profiler
    profiler = PatternProfiler(
        field_name="employee_id",
        input_file=data,
        detect_regex_patterns=True,
        common_patterns=custom_patterns
    )

    # Run profiling
    results = profiler.profile()

    # Print key results
    print(f"Total values: {results['total_values']}")
    print("\nDetected patterns:")
    for pattern, count in results['pattern_counts'].items():
        print(f"  {pattern}: {count}")

    # Create and save visualization
    output_path = profiler.visualize_results(
        output_name="employee_id_pattern_analysis"
    )
    print(f"\nVisualization saved to: {output_path}")


def example_advanced_with_dask():
    """
    Example 4: Advanced usage with Dask for large files.
    """
    print("\n=== Example 4: Advanced Usage with Dask ===")
    print("This example requires a large CSV file to demonstrate Dask capabilities.")
    print("For demonstration purposes, we'll create a sample large dataset.")

    # Create a path for our sample large file
    sample_file = Path("sample_large_data.csv")

    # Check if we need to create the sample file
    if not sample_file.exists():
        try:
            # Generate a larger dataset for demonstration
            import numpy as np

            # Define possible job titles
            job_titles = [
                "Software Engineer", "Data Scientist", "Project Manager",
                "UX Designer", "Product Manager", "QA Engineer",
                "DevOps Engineer", "Business Analyst", "Data Analyst",
                "Full Stack Developer", "Front End Developer", "Back End Developer",
                "Machine Learning Engineer", "Systems Administrator", "Network Engineer",
                "Database Administrator", "Security Engineer", "Cloud Architect",
                "Technical Writer", "IT Support Specialist"
            ]

            # Generate random data
            n_rows = 50000  # 50K rows
            np.random.seed(42)

            # Generate random job titles, some with patterns, some random
            titles = []
            for i in range(n_rows):
                if i % 10 == 0:
                    # Add some with special prefix/suffix patterns
                    titles.append(f"Senior {np.random.choice(job_titles)}")
                elif i % 11 == 0:
                    titles.append(f"{np.random.choice(job_titles)} Lead")
                elif i % 12 == 0:
                    titles.append(f"Junior {np.random.choice(job_titles)}")
                elif i % 13 == 0:
                    titles.append(f"{np.random.choice(job_titles)} Manager")
                else:
                    titles.append(np.random.choice(job_titles))

            # Create DataFrame
            large_df = pd.DataFrame({
                'job_title': titles
            })

            # Save to CSV
            large_df.to_csv(sample_file, index=False)
            print(f"Created sample large dataset with {n_rows} rows")
        except Exception as e:
            print(f"Error creating sample file: {e}")
            return

    # Configure for Dask usage
    use_dask = True
    chunk_size = 10000

    # Custom patterns for job titles
    job_patterns = [
        {"name": "senior_role", "regex": r"^Senior\s.+"},
        {"name": "junior_role", "regex": r"^Junior\s.+"},
        {"name": "lead_role", "regex": r".+\sLead$"},
        {"name": "manager_role", "regex": r".+\sManager$"}
    ]

    # Create profiler with Dask settings
    profiler = PatternProfiler(
        field_name="job_title",
        input_file=str(sample_file),
        detect_regex_patterns=True,
        common_patterns=job_patterns,
        use_dask=use_dask,
        chunk_size=chunk_size
    )

    print(f"Processing file {sample_file} using Dask (chunk_size={chunk_size})...")

    # Run profiling
    results = profiler.profile()

    # Print key results
    print(f"\nTotal job titles analyzed: {results['total_values']}")
    print("\nDetected job title patterns:")
    for pattern, count in results['most_common_patterns'].items():
        print(f"  {pattern}: {count}")

    print("\nMost common job titles:")
    for title, count in list(results.get('most_common_values', {}).items())[:5]:
        print(f"  {title}: {count}")

    # Print performance metrics
    if 'performance' in results:
        perf = results['performance']
        print("\nPerformance metrics:")
        print(f"  Total processing time: {perf.get('total_processing_time', 0):.2f} seconds")
        print(f"  Average chunk time: {perf.get('avg_chunk_time', 0):.4f} seconds")

    # Create and save visualization
    output_path = profiler.visualize_results(
        output_name="job_title_pattern_analysis"
    )
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    # Configure logging
    configure_logging(level="INFO")

    # Run examples
    try:
        example_basic_pattern_analysis()
        example_multi_value_field_analysis()
        example_custom_pattern_detection()
        example_advanced_with_dask()
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()

    print("\nExamples complete.")