"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core.fake_data
Type: Public API Package

Author: Realm Inveo Inc. & DGT Network Inc.
"""

__all__ = [
    "FakeEmailOperation",
    "FakeNameOperation",
    "FakeOrganizationOperation",
    "FakePhoneOperation",
]

from pamola_core.fake_data.operations.email_op import FakeEmailOperation

from pamola_core.fake_data.operations.name_op import FakeNameOperation

from pamola_core.fake_data.operations.organization_op import FakeOrganizationOperation

from pamola_core.fake_data.operations.phone_op import FakePhoneOperation

