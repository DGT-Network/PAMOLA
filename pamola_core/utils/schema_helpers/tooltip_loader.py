# pamola_core/utils/tooltip_loader.py
import json
from pathlib import Path

_tooltip_cache = {}

def get_tooltip(path: Path) -> dict:
    if not path.exists():
        path = Path(__file__).parent.parent / "tooltips" / path.name
    key = str(path.resolve())
    if key not in _tooltip_cache:
        with open(path, encoding="utf-8") as f:
            _tooltip_cache[key] = json.load(f)
    return _tooltip_cache[key]
