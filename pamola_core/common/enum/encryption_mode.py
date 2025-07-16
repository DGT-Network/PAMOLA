from enum import Enum


class EncryptionMode(Enum):
    """
    Encryption modes supported by the task framework.

    - NONE: No encryption
    - SIMPLE: Simple symmetric encryption
    - AGE: Age encryption (more secure, supports key rotation)
    """
    NONE = "none"
    SIMPLE = "simple"
    AGE = "age"

    @classmethod
    def from_string(cls, value: str) -> 'EncryptionMode':
        """Convert string to EncryptionMode enum value."""
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            return cls.SIMPLE