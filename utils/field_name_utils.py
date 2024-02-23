# GENERATE NEW FIELD/ATTRIBUTE THAT IS BASED ON SOME FIELD GROUPS
def generate_field_name(fields, prefix="new_", letters_count=2, max_length=12):
    """
    Generates a new attribute name based on the provided fields, prefix, and number of letters from each field.

    :param fields: A string of field names separated by semicolons (;).
    :param prefix: A prefix to be added to the beginning of the new attribute name.
    :param letters_count: Number of letters to take from each field name.
    :param max_length: The maximum length of the generated attribute name.
    :return: A new attribute name based on the specified parameters.
    """
    # Splitting the input string into a list of field names
    field_names = fields.split(';')

    # Concatenating the prefix and the specified number of letters from each field name
    new_name = prefix + ''.join([field[:letters_count] for field in field_names])

    # Ensuring the new name does not exceed the maximum length
    return new_name[:max_length]


# Usage
fields_example = "SEX;AGE;INCOME;BRAND"
new_field_name = generate_field_name(fields_example, prefix="C_", letters_count=2)
print("New Attribute Name:", new_field_name)