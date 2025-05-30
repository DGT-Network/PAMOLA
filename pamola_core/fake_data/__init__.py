"""
Initialize the fake data operations package.
"""

# Import operations to register them
from pamola_core.fake_data.operations.name_op import FakeNameOperation
from pamola_core.fake_data.operations.email_op import FakeEmailOperation
from pamola_core.fake_data.operations.organization_op import FakeOrganizationOperation
from pamola_core.fake_data.operations.phone_op import FakePhoneOperation

# Make operations available at package level
__all__ = [
    'FakeNameOperation', 
    'FakeEmailOperation', 
    'FakeOrganizationOperation', 
    'FakePhoneOperation'
    ]
