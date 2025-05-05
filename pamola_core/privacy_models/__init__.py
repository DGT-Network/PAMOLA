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

Package: Privacy Models
------------------------
This package provides implementations of formal **anonymization models** used
in anonymization and anonymization-preserving data transformations. These
models define mathematical and statistical approaches to ensure
data anonymization guarantees.

Supported anonymization models:
- **k-Anonymity**: Ensures each individual is indistinguishable from at least `k-1` others.
- **l-Diversity**: Extends k-Anonymity by ensuring diversity of sensitive attributes.
- **t-Closeness**: Ensures the distribution of sensitive attributes in a group is similar to the overall dataset.
- **k-Map**: Defines anonymization based on real-world external data mapping.

These models are critical in anonymization-preserving data sharing,
ensuring compliance with regulations such as **GDPR, HIPAA, and CCPA**.

NOTE: This module requires `pandas` and `numpy` as dependencies.

Author: Realm Inveo Inc. & DGT Network Inc.
"""
