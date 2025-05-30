"""
Cryptographic providers package for PAMOLA CORE.

This package contains all the encryption providers that implement
different encryption modes for the PAMOLA CORE cryptographic subsystem.
"""

# We'll remove the direct import of register_provider to avoid circular imports
# Instead, registration will be done externally after imports are complete

# DO NOT import router functions here to avoid circular imports
from pamola_core.utils.crypto_helpers.providers.none_provider import NoneProvider
from pamola_core.utils.crypto_helpers.providers.simple_provider import SimpleProvider
from pamola_core.utils.crypto_helpers.providers.age_provider import AgeProvider

# List all available providers for external registration
AVAILABLE_PROVIDERS = [
    NoneProvider,
    SimpleProvider,
    AgeProvider
]