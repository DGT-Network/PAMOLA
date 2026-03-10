"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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


class FakeNameOperationTooltip:
    language = (
        "• What it does: Selects the appropriate set of names to use for generation.\n"
        "• Examples: Supported built-in dictionaries include `en` (English), `ru` (Russian), and `vi` (Vietnamese).\n"
        "• Impact: This is the most important setting for ensuring the generated names are culturally realistic.\n"
        "• Default: `en`"
    )

    format = (
        "• How it works: Uses placeholders for First (F), Last (L), and Middle (M) names.\n"
        "• Available Formats: `FL` (John Smith), `LF` (Smith John), `FML` (John Michael Smith), `LFM` (Smith John Michael), `F_L` (John_Smith), `L_F` (Smith_John).\n"
        '• Note: You can control casing directly via the format. Using `fml` will make the output lowercase, and `FML_` will make it uppercase, overriding the global "Name Case" setting.\n'
        "• Default: `FL`"
    )

    case = (
        "• What it does: Controls the text case of the output.\n"
        "• Options: `title` (John Smith), `upper` (JOHN SMITH), `lower` (john smith).\n"
        '• Note: This setting can be overridden by using a case-specific format string in the "Name Format" field (e.g., `fml`).\n'
        "• Recommended: `title` for most use cases.\n"
        "• Default: `title`"
    )

    use_faker = (
        "• What it does: Switches from the built-in name dictionaries to the popular Faker data generation library.\n"
        "• Trade-offs:\n"
        "- Pros: Provides a much larger and more diverse pool of names.\n"
        "- Cons: Requires the Faker library to be installed and may have a slight performance overhead.\n"
        "• Default: Disabled"
    )

    dictionaries = (
        "• How it works: Provide a JSON object specifying the language and name type, with a value pointing to your file path.\n"
        '• Example: `{"en": {"male_first_names": "/path/to/en_m_first.txt", "last_names": "/path/to/en_last.txt"}}`\n'
        "• Impact: Gives you full control over the pool of names. This is an advanced feature."
    )

    gender_from_name = (
        '• How it works: The system will analyze the original first name (e.g., "Mary") and determine its likely gender ("Female") before generating a new female name.\n'
        "• Impact: Useful when you don't have a separate gender column but want to maintain gender accuracy.\n"
        "• Note: The `Gender Field` setting, if configured, will always take precedence over this option.\n"
        "• Default: Disabled"
    )

    gender_field = (
        "• What it does: Tells the operation where to find the gender for each person, ensuring a male name is replaced with another male name.\n"
        '• Example: Select a column named "Gender". The system recognizes values like `M`, `Male`, `1` for male and `F`, `Female`, `2` for female.\n'
        "• Impact: This is the most reliable way to ensure gender-correct name generation."
    )

    f_m_ratio = (
        "• How it works: A slider or input from 0.0 to 1.0.\n"
        "• Example: `0.5` means a 50% chance of generating a female name. `0.7` means a 70% chance.\n"
        "• Impact: Controls the gender balance in the generated data when no gender information is provided.\n"
        "• Default: `0.5`"
    )

    consistency_mechanism = (
        '• What it does: Guarantees that "John Doe" is always replaced by "Mark Smith" every time the operation runs.\n'
        "• Options:\n"
        "- `prgn`: A fast, stateless method using a cryptographic key. Best for performance.\n"
        "- `mapping`: A stateful method that stores every original-to-fake pair. Slower but allows you to save and reuse the exact same mappings.\n"
        "• Recommended: `prgn` for most scenarios.\n"
        "• Default: `prgn`"
    )

    id_field = (
        "• What it does: Provides a stable identifier for each record to the generation logic.\n"
        "• Note: In the current `NameGenerator` implementation, consistency is primarily based on the original name value itself, not this ID field. Therefore, it will not help resolve name typos into a single fake identity.\n"
        "• Impact: Limited for this specific operation, but may be used by other generator types."
    )

    key = (
        "• What it does: Acts like a password for the generation algorithm. The same key with the same input name will always produce the same output name.\n"
        "• Example: `my-secret-project-key-2025`\n"
        "• Impact: Crucial for consistency. If you lose the key, you cannot reproduce the same fake names again."
    )

    context_salt = (
        "• How it works: It's mixed with the key during generation.\n"
        '• Example: Use `dev` for development and `prod` for production. "John Doe" with the same key will become "Mark Smith" in `dev` but "Peter Jones" in `prod`.\n'
        "• Impact: Allows you to create multiple, distinct pseudonymized datasets from a single source."
    )

    mapping_store_path = (
        "• What it does: Specifies a file (e.g., C:\\mappings\\customer_names.json) that stores the lookup table.\n"
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
        "• Example: `fake_customer_name`"
    )

    column_prefix = (
        "• How it works: If your original column is customer_name and the prefix is fake_, the new column will be named fake_customer_name.\n"
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

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "language": cls.language,
            "format": cls.format,
            "case": cls.case,
            "use_faker": cls.use_faker,
            "dictionaries": cls.dictionaries,
            "gender_from_name": cls.gender_from_name,
            "gender_field": cls.gender_field,
            "f_m_ratio": cls.f_m_ratio,
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
        }
