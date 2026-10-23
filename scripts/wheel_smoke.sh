#!/usr/bin/env bash
# Install the EXACT built wheel into a brand-new virtualenv and prove it works.
#
#   scripts/wheel_smoke.sh <dist-dir> <expected-version>
#
# Used by ci.yml (pull requests) and release.yml (every release candidate), so the
# artifact that gets published is the artifact that was tested. Needs bash (present
# on every GitHub-hosted runner, Windows included) and a python on PATH.
set -euo pipefail

DIST_DIR="${1:?usage: wheel_smoke.sh <dist-dir> <expected-version>}"
EXPECTED_VERSION="${2:?usage: wheel_smoke.sh <dist-dir> <expected-version>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="$(cd "$DIST_DIR" && pwd)"

shopt -s nullglob
wheels=("$DIST_DIR"/*.whl)
if [ "${#wheels[@]}" -ne 1 ]; then
  echo "ERROR: expected exactly one wheel in $DIST_DIR, found ${#wheels[@]}" >&2
  exit 1
fi
WHEEL="${wheels[0]}"
echo "wheel: $WHEEL"
case "$(basename "$WHEEL")" in
  "pamola_core-${EXPECTED_VERSION}-"*) ;;
  *) echo "ERROR: wheel name does not carry version $EXPECTED_VERSION" >&2; exit 1 ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python -m venv "$WORK/venv"
if [ -d "$WORK/venv/Scripts" ]; then BIN="$WORK/venv/Scripts"; else BIN="$WORK/venv/bin"; fi

# Only the wheel and what it declares: no -e, no extras, no checkout on the path.
"$BIN/python" -m pip install --quiet --upgrade pip
"$BIN/python" -m pip install --quiet "$WHEEL"
"$BIN/python" -m pip check

# Run from an empty directory so a checkout can never shadow the installed package.
cd "$WORK"
"$BIN/python" -c "import pamola_core; print('pamola_core', pamola_core.__version__, pamola_core.__file__)"
"$BIN/pamola-core" --version | tee "$WORK/version.txt"
grep -q "$EXPECTED_VERSION" "$WORK/version.txt"
"$BIN/pamola-core" list-ops >/dev/null
echo "CLI ok"
"$BIN/python" "$SCRIPT_DIR/clean_wheel_smoke.py" --expect-version "$EXPECTED_VERSION"
