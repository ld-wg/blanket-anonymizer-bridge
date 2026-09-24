#!/usr/bin/env bash
# One-shot setup for the two isolated venvs this bridge needs.
#
# Python >=3.10 is REQUIRED, not just BLANKET's own looser declared
# `>=3.9` (its vendored FaceFusion hard-checks `sys.version_info < (3, 10)`
# at runtime — see NOTICE.md). Pass an explicit interpreter as $1 if the
# default `python3` on this machine is older; on serra1, `python3.12` is
# already the system default and is what this script uses.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${1:-python3.12}"

echo "== checking $PYTHON version =="
"$PYTHON" -c "import sys; assert sys.version_info >= (3, 10), f'{sys.version} is too old — BLANKET\'s own vendored FaceFusion needs >=3.10'; print(sys.version)"

echo "== initializing vendor/blanket-infant-face-anonym submodule =="
git -C "$HERE" submodule update --init --recursive

echo "== creating .venv-identity =="
"$PYTHON" -m venv "$HERE/.venv-identity"
"$HERE/.venv-identity/bin/pip" install --upgrade pip
"$HERE/.venv-identity/bin/pip" install -r "$HERE/requirements-identity.txt"

echo "== creating .venv-swap =="
"$PYTHON" -m venv "$HERE/.venv-swap"
"$HERE/.venv-swap/bin/pip" install --upgrade pip
"$HERE/.venv-swap/bin/pip" install -r "$HERE/requirements-swap.txt"

echo "== done =="
echo "identity interpreter: $HERE/.venv-identity/bin/python"
echo "swap interpreter:     $HERE/.venv-swap/bin/python"
echo "Point lose-the-faces-keep-the-lesson's --blanket-repo at: $HERE"
