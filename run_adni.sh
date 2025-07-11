#!/bin/bash
set -e  # Exit immediately on error

echo "run_adni.sh started at $(date)"
echo "Running in directory: $(pwd)"
echo "Running with arguments: $@"

# Prevent user-level site packages from interfering
export PYTHONNOUSERSITE=1

# ==============================================================================
# 📂 Prepare directories
# ==============================================================================
mkdir -p adni_logs
echo "Created logs directory at $(pwd)/adni_logs"

mkdir -p "adni_results"

# ==============================================================================
# 🐍 Conda Env Extraction
# ==============================================================================
ENV_TARBALL="env.tar.gz"
ENV_DIR=".conda_env"
# MINICONDA_SH="miniconda.sh"
# MINICONDA_DIR="$PWD/miniconda"
PYTHON_EXEC=""

rm -rf "$ENV_DIR" "$MINICONDA_DIR"

if [[ -f "$ENV_TARBALL" ]]; then
    echo "Extracting environment..."
    mkdir -p "$ENV_DIR"
    tar -xzf "$ENV_TARBALL" -C "$ENV_DIR"
    PYTHON_EXEC="$ENV_DIR/bin/python"
    echo "Using extracted environment at $PYTHON_EXEC"
else
    echo "⚠️ env.tar.gz not found. Will attempt Miniconda setup if needed."
fi


# ==============================================================================
# 🧪 Final sanity check
# ==============================================================================
echo "=== ENVIRONMENT VALIDATION ==="
echo "Python path: $PYTHON_EXEC"
echo "Python version: $($PYTHON_EXEC --version)"

if ! "$PYTHON_EXEC" -c "import kde_ebm, pyebm, pysaebm, yaml; import scipy._lib" &>/dev/null; then
    echo "❌ Final environment validation failed — aborting"
    exit 1
fi

# ==============================================================================
# ▶️ Run Python Script
# ==============================================================================
echo "=== STARTING MAIN SCRIPT ==="
"$PYTHON_EXEC" ./run_adni.py "$@"

echo "✅ Script completed at $(date)"
