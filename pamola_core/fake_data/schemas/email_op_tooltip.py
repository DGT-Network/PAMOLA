
class FakeEmailOperationTooltip:
    format = (
        "• What it does: Forces all generated emails to follow one specific pattern.\n"
        "    • Options:\n"
        "    - `name_surname`: e.g., `john.smith`\n"
        "    - `surname_name`: e.g., `smith.john`\n"
        "    - `nickname`: e.g., `cooluser123`\n"
        "    • Note: If this is left blank, the format will be chosen randomly based on the 'Format Ratio'."
    )
    format_ratio = (
        "• How it works: Provide a JSON object with formats and their desired weights.\n"
        "    • Example: {'name_surname': 0.5, 'nickname': 0.5} will generate name-based emails 50% of the time and nickname-based emails 50% of the time.\n"
        "    • Impact: Allows you to control the mix of email styles in the final output. The weights should add up to 1.0."
    )
    separator_options = (
        "• What it does: Controls the separator between first and last names (e.g., . in john.smith).\n"
        "    • Example: A list like ['.', '_', ''] allows the generator to produce john.smith, john_smith, or johnsmith.\n"
        "    • Default: ['.', '_', '-', '']"
    )
    number_suffix_probability = (
        "• How it works: A value of 0.4 means there is a 40% chance of adding a number.\n"
        "    • Example: john.smith might become john.smith842.\n"
        "    • Impact: Increases the uniqueness and realism of the generated emails.\n"
        "    • Default: 0.4 (40%)"
    )
    preserve_domain_ratio = (
        "• How it works: If set to 0.8, there is an 80% chance that an email from user@company.com will be replaced by a new email that also ends in @company.com.\n"
        "    • Impact: Higher values preserve the original domain distribution, which is good for utility but may be less private.\n"
        "    • Default: 0.5 (50%)"
    )
    business_domain_ratio = (
        "• How it works: A value of 0.2 means a 20% chance of generating a domain like @acme-corp.com instead of a common one like @gmail.com.\n"
        "    • Impact: Helps simulate a mix of professional and personal email addresses.\n"
        "    • Default: 0.2 (20%)"
    )
    max_length = (
        "• What it does: Ensures that generated emails do not exceed standard length limits.\n"
        "    • Impact: The local part (before the '@') will be truncated if the total length exceeds this value.\n"
        "    • Recommended: Keep the default unless you have specific requirements.\n"
        "    • Default: 254"
    )
    first_name_field = (
        "• What it does: Provides the 'first name' component for formats like name_surname.\n"
        "    • Example: Select a column named 'FirstName'.\n"
        "    • Impact: Enables the creation of highly realistic emails like john.smith@example.com."
    )
    last_name_field = (
        "• What it does: Provides the 'last name' component for formats like name_surname.\n"
        "    • Example: Select a column named 'LastName'."
    )
    full_name_field = (
        "• What it does: If you have a single column with the full name, this operation can split it into first and last names for email generation.\n"
        "    • Example: Select a column named 'FullName'.\n"
        "    • Note: Use the 'Name Format' parameter to tell the system how to parse the full name correctly."
    )
    name_format = (
        "• How it works: Tells the parser the order of name parts.\n"
        "    • Examples: FL (for 'John Smith'), LF (for 'Smith, John').\n"
        "    • Impact: Essential for correctly using the full_name_field."
    )
    nicknames_dict = (
        "• What it does: Overrides the built-in list of generic nicknames (like 'cooluser', 'digitalmind').\n"
        "    • Example: A file with one nickname per line (e.g., gamer1, test_user, etc.).\n"
        "    • Impact: Allows you to tailor the generated nicknames to a specific context."
    )
    domains = (
        "• How it works: Provides the pool of domains (e.g., my-company.com, my-startup.io) to be used when generating a new email domain.\n"
        "    • Impact: Overrides the default list of common public domains (gmail, yahoo, etc.). This gives you full control over the generated email domains."
    )
    consistency_mechanism = (
        "• What it does: Guarantees that 'john.doe@example.com' is always replaced by 'mark.smith@new.com' every time the operation runs.\n"
        "    • Options:\n"
        "    - prgn: A fast, stateless method using a cryptographic key. Best for performance.\n"
        "    - mapping: A stateful method that stores every original-to-fake pair. Slower but allows you to save and reuse the exact same mappings.\n"
        "    • Recommended: prgn for most scenarios.\n"
        "    • Default: prgn"
    )
    id_field = (
        "• What it does: Provides a stable identifier for each record to the generation logic.\n"
        "    • Note: The EmailGenerator's logic primarily uses the original email value itself to ensure consistency. Therefore, this parameter has limited impact for this specific operation."
    )
    key = (
        "• What it does: Acts like a password for the generation algorithm. The same key with the same input email will always produce the same output email.\n"
        "    • Example: my-secret-project-key-2025\n"
        "    • Impact: Crucial for consistency. If you lose the key, you cannot reproduce the same fake emails again."
    )
    context_salt = (
        "• How it works: It's mixed with the key during generation.\n"
        "    • Example: Use dev for development and prod for production. The same email and key will generate different results in the two environments.\n"
        "    • Impact: Allows you to create multiple, distinct pseudonymized datasets from a single source.\n"
        "    • Default: email-generation"
    )
    mapping_store_path = (
        "• What it does: Specifies a file (e.g., C:\\mappings\\email_map.json) that stores the lookup table.\n"
        "    • Impact: If the file exists, the operation will reuse its mappings. If not, a new map can be created and saved there."
    )
    save_mapping = (
        "• What it does: Writes the complete list of original emails and their corresponding fake emails to a file after the operation finishes.\n"
        "    • Impact: Enables perfect consistency across future runs and allows for auditing, but the saved file will contain sensitive original data and must be secured."
    )
    validate_source = (
        "• What it does: Helps identify and handle malformed email addresses in your data.\n"
        "    • Impact: This setting determines whether the 'Invalid Email Handling' rule is applied.\n"
        "    • Default: Enabled"
    )
    handle_invalid_email = (
        "• Options:\n"
        "    - generate_new: Replaces the invalid email with a completely new, valid one.\n"
        "    - keep_empty: Replaces the invalid email with a blank value.\n"
        "    - generate_with_default_domain: Creates a new email but uses a known, safe domain.\n"
        "    • Recommended: generate_new to ensure all rows have data.\n"
        "    • Default: generate_new"
    )
    mode = (
        "• Options:\n"
        "    - REPLACE: Overwrites the original email column with fake emails. This is a destructive action.\n"
        "    - ENRICH: Keeps the original column and adds a new column containing the fake emails.\n"
        "    • Recommended: ENRICH for safety and data validation."
    )
    output_field_name = (
        "• How it works: If you leave this blank, a name will be automatically created using the Column Prefix and the original field name.\n"
        "    • Example: fake_email_address"
    )
    column_prefix = (
        "• How it works: If your original column is email and the prefix is fake_, the new column will be named fake_email.\n"
        "    • Example: fake_, anon_, synthetic_"
    )
    null_strategy = (
        "• Options:\n"
        "    - preserve: Keeps null values as they are (default).\n"
        "    - replace: Generates a new fake email to fill in the blank.\n"
        "    • Recommended: preserve to maintain the original data's completeness characteristics."
    )
    force_recalculation = (
        "• What it does: Ignores any previously saved results (cache) for this exact operation and re-generates all data from scratch.\n"
        "    • Impact: Use this if you have changed a setting or the underlying data and need to ensure the results are fresh."
    )
    detailed_metrics = (
        "• What it does: Provides deeper insights into the operation's output.\n"
        "    • Impact: May slightly slow down processing due to the extra data collection.\n"
        "    • Default: Disabled"
    )
    max_retries = (
        "• What it does: Increases the operation's resilience against random generation errors.\n"
        "    • Impact: A higher value can overcome intermittent issues but might slow down the process if there is a persistent problem.\n"
        "    • Default: 3"
    )

    @classmethod
    def as_dict(cls):
        """Return tooltips as a dictionary for Formily or schema builders."""
        return {
            "format": cls.format,
            "format_ratio": cls.format_ratio,
            "separator_options": cls.separator_options,
            "number_suffix_probability": cls.number_suffix_probability,
            "preserve_domain_ratio": cls.preserve_domain_ratio,
            "business_domain_ratio": cls.business_domain_ratio,
            "max_length": cls.max_length,
            "first_name_field": cls.first_name_field,
            "last_name_field": cls.last_name_field,
            "full_name_field": cls.full_name_field,
            "name_format": cls.name_format,
            "nicknames_dict": cls.nicknames_dict,
            "domains": cls.domains,
            "consistency_mechanism": cls.consistency_mechanism,
            "id_field": cls.id_field,
            "key": cls.key,
            "context_salt": cls.context_salt,
            "mapping_store_path": cls.mapping_store_path,
            "save_mapping": cls.save_mapping,
            "validate_source": cls.validate_source,
            "handle_invalid_email": cls.handle_invalid_email,
            "mode": cls.mode,
            "output_field_name": cls.output_field_name,
            "column_prefix": cls.column_prefix,
            "null_strategy": cls.null_strategy,
            "force_recalculation": cls.force_recalculation,
            "detailed_metrics": cls.detailed_metrics,
            "max_retries": cls.max_retries,
        }
