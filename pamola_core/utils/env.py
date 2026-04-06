"""
Environment loading utilities.

Centralizes dotenv handling to avoid import-time side effects across modules.
"""

from dotenv import load_dotenv

_ENV_LOADED = False


def load_dotenv_once() -> None:
    """
    Load environment variables from a .env file once per process.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv()
    _ENV_LOADED = True
