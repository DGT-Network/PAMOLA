"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
This file is part of the PAMOLA ecosystem, a comprehensive suite for
anonymization-enhancing technologies. PAMOLA.CORE serves as the open-source
foundation for anonymization-preserving data processing.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Package: pamola_core.attacks (experimental - API may change)
Type: Internal package. Not exported from pamola_core.__init__.

Author: Realm Inveo Inc. & DGT Network Inc.

Scope
-----
This package carries only the attacks that measure **anonymization** quality:

* ``LinkageAttack``       - record linkage, including the CVPL variant
                            (TruncatedSVD + cosine similarity)
* ``AttributeInference``  - attribute disclosure from quasi-identifiers
* ``SinglingOutAttack``   - isolation of a single record via quasi-identifiers
                            (worst-case subset sweep + SUDA2/MSU attribution)

Together these cover the three EDPB anonymisation criteria: singling-out,
linkability and inference.

Three modules were removed in the 1.0 cleanup:

* ``membership_inference`` - membership inference is a model-privacy question,
  not an anonymization one; it belongs to the SYNT/BEST track.
* ``distance_to_closest_record`` and ``nearest_neighbor_distance_ratio`` -
  these duplicated ``pamola_core.metrics.privacy.distance`` and
  ``pamola_core.metrics.privacy.neighbor``, which are the *live*
  implementations: ``PrivacyMetricOperation`` imports the metrics versions and
  has never imported these. The copies here were dead code (77 and 76 lines
  against 365 and 149).

DCR and NNDR remain fully available - import them from
``pamola_core.metrics.privacy``.

Singling-out
------------
``singling_out`` provides two complementary views. The sweep answers "how
exposed is this dataset" (worst-case quasi-identifier subset, with an
independent-marginals baseline so the rate is interpretable). The SUDA2/MSU
engine answers "which columns cause it", by finding the minimal attribute
combinations that isolate each record and ranking attributes by contribution.

The sweep and baseline are ported from PAMOLA.BEST ``ATK-SINGLING-001``
(polars to pandas); the MSU/SUDA2 engine from the clean-room implementation in
PAMOLA spikes. No GPL sources were used.
"""

__all__ = [
    # Base
    "AttackInitialization",
    "BaseAttack",
    # Anonymization-facing attacks
    "LinkageAttack",
    "AttributeInference",
    "AttributeInferenceAttack",
    "SinglingOutAttack",
    # Singling-out primitives
    "singling_out_sweep",
    "independent_marginals_baseline",
    "minimal_sample_uniques",
    "suda_scores",
    "dis_risk",
    "enumerate_subset_count",
    "wilson_ci",
    "RecordRisk",
    # Metrics
    "AttackMetrics",
]

from pamola_core.attacks.base import AttackInitialization
from pamola_core.attacks.base import AttackInitialization as BaseAttack

from pamola_core.attacks.linkage_attack import LinkageAttack

from pamola_core.attacks.attribute_inference import AttributeInference
from pamola_core.attacks.attribute_inference import AttributeInference as AttributeInferenceAttack

from pamola_core.attacks.singling_out import (
    RecordRisk,
    SinglingOutAttack,
    dis_risk,
    enumerate_subset_count,
    independent_marginals_baseline,
    minimal_sample_uniques,
    singling_out_sweep,
    suda_scores,
    wilson_ci,
)

from pamola_core.attacks.attack_metrics import AttackMetrics
