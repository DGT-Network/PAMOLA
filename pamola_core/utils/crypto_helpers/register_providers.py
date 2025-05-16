"""
Register providers for the PAMOLA cryptographic subsystem.

This module registers all the available crypto providers with the router.
It is kept separate to avoid circular imports.
"""

import os
from pamola_core.utils.io_helpers.crypto_router import register_provider
from pamola_core.utils.crypto_helpers.providers import AVAILABLE_PROVIDERS

# Lazy registration of providers to avoid initialization issues
PROVIDER_CLASSES = []


def register_all_providers():
    """
    Register all available crypto providers with the router.

    This function now implements lazy loading based on environment variables,
    allowing selective enabling/disabling of providers.
    """
    global PROVIDER_CLASSES

    # Skip if already registered
    if PROVIDER_CLASSES:
        return

    # Allow env-var white/black list
    enabled = os.getenv("PAMOLA_SUPPORTED_CRYPTO_PROVIDERS", "all")

    # Process AVAILABLE_PROVIDERS, filtering as needed
    for provider_class in AVAILABLE_PROVIDERS:
        provider_name = provider_class.__name__.lower().replace('provider', '')

        # Check if this provider should be enabled
        if enabled.lower() == "all" or provider_name in enabled.lower():
            # Special handling for AgeProvider to avoid immediate check
            if provider_name == "age" and os.getenv("PAMOLA_DISABLE_AGE", "0") == "1":
                continue

            PROVIDER_CLASSES.append(provider_class)

    # Register all enabled providers
    for provider_class in PROVIDER_CLASSES:
        try:
            register_provider(provider_class)
        except Exception as e:
            # Log the error but continue with other providers
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to register provider {provider_class.__name__}: {str(e)}"
            )