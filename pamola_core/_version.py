# pamola_core/_version.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pamola-core")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"