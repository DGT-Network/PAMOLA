"""
PAMOLA.CORE - Regex patterns Utilities
---------------------------------------------------  
This module contains common regular expressions (regex) for general usage.

Features:
 - This module defines commonly used regex patterns.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""
class Patterns:
    EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE_REGEX = r"^\+?(?:\d[\d\-. ]+)?(?:\([\d\-. ]+\))?[\d\-. ]+\d$"
    CREDIT_CARD = r"^\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}$"
    DOMAIN_REGEX = r"@([\w.-]+)"
    PCI_PATTERNS = r'^\d{16}$|^(\d{4}[ ]{1}){3}\d{4}$|^(\d{4}[-]{1}){3}\d{4}$'
    CREDIT_PATTERNS = r'^\d{13,16}$|^(\d{4}[ ]{1}){3}\d{1,4}$|^(\d{4}[-]{1}){3}\d{1,4}$'