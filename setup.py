# setup.py
from pathlib import Path
import tomllib
from setuptools import setup

def update_version():
    pyproject_file = Path(__file__).parent / "pyproject.toml"
    version_file = Path(__file__).parent / "pamola_core/_version.py"

    with pyproject_file.open("rb") as f:
        pyproject = tomllib.load(f)

    version = pyproject["project"]["version"]
    version_file.write_text(f'__version__ = "{version}"\n')
    print(f"Updated {version_file} to version {version}")

update_version()
setup()