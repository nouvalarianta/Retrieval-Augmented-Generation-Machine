#!/usr/bin/env bash
# Menyiapkan virtual environment Python dan menginstal dependensi Machine Assistant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

echo "[INFO] Menggunakan Python: $($PYTHON_BIN --version)"

if [ ! -d ".venv" ]; then
    echo "[INFO] Membuat virtual environment di .venv..."
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements/base.txt
python -m pip install -e .

# Memasang pre-commit hook jika repositori git aktif
if [ -d ".git" ]; then
    cp scripts/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "[INFO] Pre-commit hook terpasang."
fi

# Memastikan direktori operasional tersedia
mkdir -p docs index debug

echo "[INFO] Setup selesai!"
echo "[INFO] Uji sistem dengan perintah:"
echo "       source .venv/bin/activate && ma ingest --rebuild"
