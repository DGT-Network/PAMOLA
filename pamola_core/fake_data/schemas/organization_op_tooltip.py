"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Name Operation Tooltips
Package:       pamola_core.anonymization.schemas
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-15
License:       BSD 3-Clause

Description:
Provides detailed tooltips for name generation configuration fields in PAMOLA.CORE.
- Explains language, format, case, gender, consistency, and output options for name anonymization
- Designed for integration with Formily and schema-driven UI builders
- Improves user understanding and correct configuration of name generation operations

Changelog:
1.0.0 - 2025-01-15 - Initial creation of name operation tooltip file
"""


class FakeOrganizationOperationTooltip:
    organization_type = (
        "• What it does: Selects the primary dictionary of names to use.\n"
        "• Options: `general`, `educational`, `manufacturing`, `government`, `industry`.\n"
        '• Note: If you select `industry`, you must also specify which industry in the "Industry Type" field.\n'
        "• Default: `general`"
    )

    region = (
        "• What it does: Ensures that names, prefixes, and suffixes are culturally appropriate.\n"
        '• Example: `en` for English names like "Global Inc.", `ru` for Russian names like "ООО Глобал".\n'
        "• Default: `en`"
    )

    industry = (
        "• What it does: Narrows down the name generation to a specific business sector for more relevant results.\n"
        "• Examples: `tech`, `finance`, `retail`, `healthcare`.\n"
        '• Impact: Creates more contextually specific names like "Quantum Software" instead of a generic "Global Group".'
    )

    preserve_type = (
        "• How it works: An original name like \"State University\" will be identified as 'educational' and replaced with another educational institution name.\n"
        "• Impact: Helps maintain the semantic characteristics of your original data.\n"
        "• Default: Enabled"
    )

    add_prefix_probability = (
        "• How it works: A value of 0.3 means there is a 30% chance of adding a prefix.\n"
        '• Example: "Stellar Solutions" might become "International Stellar Solutions".\n'
        "• Impact: Adds variation to the generated names.\n"
        "• Default: `0.3` (30%)"
    )

    add_suffix_probability = (
        "• How it works: A value of 0.5 means there is a 50% chance of adding a suffix.\n"
        '• Example: "Apex Digital" might become "Apex Digital LLC".\n'
        "• Impact: Makes the generated names appear more formal and realistic.\n"
        "• Default: `0.5` (50%)"
    )

    type_field = (
        '• What it does: Overrides the "Default Organization Type" on a per-row basis, allowing for dynamic generation.\n'
        "• Example: If a row has 'educational' in this column, an educational name will be generated for that row, regardless of the default setting.\n"
        "• Impact: Powerful for datasets with mixed organization types."
    )

    region_field = (
        '• What it does: Overrides the "Default Region" setting on a per-row basis.\n'
        "• Example: If a row has 'ru' in this column, a Russian-style organization name will be generated for that specific row.\n"
        "• Impact: Enables highly accurate, row-level regional name generation."
    )

    dictionaries = (
        "• How it works: Provide a JSON object linking an organization type to a file path.\n"
        '• Example: `{"general": "/path/to/my_companies.txt", "educational": "/path/to/schools.txt"}`\n'
        "• Impact: Gives you full control over the pool of base names used for generation."
    )

    prefixes = (
        "• How it works: Provide a JSON object linking an organization type to a file of prefixes.\n"
        '• Example: `{"tech": "/path/to/tech_prefixes.txt"}`\n'
        "• Impact: Tailors the generated prefixes to your specific business context."
    )

    suffixes = (
        "• How it works: Provide a JSON object linking an organization type to a file of suffixes.\n"
        '• Example: `{"general": "/path/to/legal_endings.txt"}`\n'
        "• Impact: Allows you to use region-specific or industry-specific legal identifiers."
    )

    consistency_mechanism = (
        '• What it does: Guarantees that "Acme Corp" is always replaced by "Stellar Inc." every time the operation runs.\n'
        "• Options:\n"
        "- `prgn`: A fast, stateless method using a cryptographic key. Best for performance.\n"
        "- `mapping`: A stateful method that stores every original-to-fake pair. Slower but allows you to save and reuse the exact same mappings.\n"
        "• Recommended: `prgn` for most scenarios.\n"
        "• Default: `prgn`"
    )

    id_field = (
        "• What it does: Provides a stable identifier for each record to the generation logic.\n"
        "• Note: In the current `OrganizationGenerator` implementation, consistency is primarily based on the original organization name itself, not this ID field."
    )

    key = (
        "• What it does: Acts like a password for the generation algorithm. The same key with the same input name will always produce the same output name.\n"
        "• Example: `my-secret-project-key-2025`\n"
        "• Impact: Crucial for consistency. If you lose the key, you cannot reproduce the same fake names again."
    )

    context_salt = (
        "• How it works: It's mixed with the key during generation.\n"
        '• Example: Use `dev` for development and `prod` for production. "Acme Corp" with the same key will generate a different fake name in each environment.\n'
        "• Impact: Allows you to create multiple, distinct pseudonymized datasets from a single source.\n"
        "• Default: `org-generation`"
    )

    mapping_store_path = (
        "• What it does: Specifies a file (e.g., C:\\mappings\\org_names.json) that stores the lookup table.\n"
        "• Impact: If the file exists, the operation will reuse its mappings. If not, a new map can be created and saved there."
    )

    save_mapping = (
        "• What it does: Writes the complete list of original names and their corresponding fake names to a file after the operation finishes.\n"
        "• Impact: Enables perfect consistency across future runs and allows for auditing, but the saved file will contain sensitive original data and must be secured."
    )

    mode = (
        "• Options:\n"
        "- `REPLACE`: Overwrites the original name column with fake names. This is a destructive action.\n"
        "- `ENRICH`: Keeps the original column and adds a new column containing the fake names.\n"
        "• Recommended: `ENRICH` for safety and data validation."
    )

    output_field_name = (
        "• How it works: If you leave this blank, a name will be automatically created using the Column Prefix and the original field name.\n"
        "• Example: `fake_company_name`"
    )

    column_prefix = (
        "• How it works: If your original column is company_name and the prefix is fake_, the new column will be named fake_company_name.\n"
        "• Example: `fake_`, `anon_`, `synthetic_`"
    )

    null_strategy = (
        "• Options:\n"
        "- `preserve`: Keeps the null values as they are (default).\n"
        "- `replace`: Generates a new fake name to fill in the blank.\n"
        "• Recommended: `preserve` to maintain the original data's completeness characteristics."
    )

    force_recalculation = (
        "• What it does: Ignores any previously saved results (cache) for this exact operation and re-generates all data from scratch.\n"
        "• Impact: Use this if you have changed a setting or the underlying data and need to ensure the results are fresh."
    )

    max_retries = (
        "• What it does: Increases the operation's resilience against random generation errors.\n"
        "• Impact: A higher value can overcome intermittent issues but might slow down the process if there is a persistent problem.\n"
        "• Default: `3`"
    )

    detailed_metrics = (
        "• What it does: Provides deeper insights into the operation's output, such as which types of names were generated and from which regions.\n"
        "• Impact: May slightly slow down processing due to the extra data collection.\n"
        "• Default: Disabled"
    )

    collect_type_distribution = (
        "• What it does: Activates a specific part of the metrics collection to analyze the distribution of types like 'educational', 'government', etc., in the output.\n"
        "• Impact: Provides valuable insight into the composition of the generated data.\n"
        "• Default: Enabled"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "organization_type": cls.organization_type,
            "region": cls.region,
            "industry": cls.industry,
            "preserve_type": cls.preserve_type,
            "add_prefix_probability": cls.add_prefix_probability,
            "add_suffix_probability": cls.add_suffix_probability,
            "type_field": cls.type_field,
            "region_field": cls.region_field,
            "dictionaries": cls.dictionaries,
            "prefixes": cls.prefixes,
            "suffixes": cls.suffixes,
            "consistency_mechanism": cls.consistency_mechanism,
            "id_field": cls.id_field,
            "key": cls.key,
            "context_salt": cls.context_salt,
            "mapping_store_path": cls.mapping_store_path,
            "save_mapping": cls.save_mapping,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
            "max_retries": cls.max_retries,
            "detailed_metrics": cls.detailed_metrics,
            "collect_type_distribution": cls.collect_type_distribution,
        }
