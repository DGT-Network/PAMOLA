"""
pamola_core/utils/tasks/task_context.py

TaskContext: carries reproducibility seed and per-operation RNG.
Implements FR-EP3-CORE-043 — reproducibility via seed propagation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TaskContext:
    """
    Carries execution context (seed, RNG, task_dir) for a running task.

    Parameters
    ----------
    seed : int, optional
        Master random seed. When set, all operations receive deterministic
        per-operation seeds derived from this value.
    task_dir : Path, optional
        Root task directory.
    """

    seed: Optional[int] = None
    task_dir: Optional[Path] = None
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def get_operation_seed(self, op_name: str) -> Optional[int]:
        """
        Return a deterministic seed for a named operation.

        Derived by hashing the master seed + operation name so that:
        - same master seed + same op_name → same operation seed
        - different ops get different seeds (no correlation)

        Parameters
        ----------
        op_name : str
            Operation class name or unique identifier.

        Returns
        -------
        int or None
            Integer seed if master seed is set, None otherwise.
        """
        if self.seed is None:
            return None
        digest = hashlib.sha256(f"{self.seed}:{op_name}".encode()).digest()
        # Use first 4 bytes as unsigned int — fits numpy/random seed range
        return int.from_bytes(digest[:4], byteorder="big")
