#!/bin/bash
set -e  # Exit immediately on error

echo "run_mlhc.sh started at $(date)"
echo "Running in directory: $(pwd)"
echo "Running with arguments: $@"

cp /staging/hhao9/data.tar.gz .

# Prevent user-level site packages from interfering
export PYTHONNOUSERSITE=1

# ==============================================================================
# 📂 Prepare directories
# ==============================================================================
mkdir -p logs
chmod 755 logs
echo "Created logs directory at $(pwd)/logs"

for dir in debm debm_gmm ucl_gmm ucl_kde conjugate_priors hard_kmeans mle em kde; do
    mkdir -p "algo_results/$dir"
done

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

# # ==============================================================================
# # ✅ Try extracted env first, fallback to Miniconda if failed
# # ==============================================================================
# if [[ -n "$PYTHON_EXEC" ]] && "$PYTHON_EXEC" -c "import kde_ebm, pyebm, pysaebm, yaml; import scipy._lib" &>/dev/null; then
#     echo "✅ Extracted environment passed dependency check"
# else
#     echo "❌ Extracted environment failed — falling back to Miniconda"

#     bash "$MINICONDA_SH" -b -p "$MINICONDA_DIR"

#     "$MINICONDA_DIR/bin/conda" create -y -p "$PWD/.conda_env" python=3.9
#     "$MINICONDA_DIR/bin/conda" install -y -p "$PWD/.conda_env" -c conda-forge pip setuptools wheel

#     "$PWD/.conda_env/bin/pip" install --no-cache-dir --isolated --no-input pyebm pysaebm pyyaml
#     "$PWD/.conda_env/bin/pip" install --no-cache-dir --isolated --no-input git+https://github.com/ucl-pond/kde_ebm

#     PYTHON_EXEC="$PWD/.conda_env/bin/python"
# fi

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
# 📦 Extract data
# ==============================================================================
if [[ -f "data.tar.gz" ]]; then
    echo "Extracting data..."
    tar -xzf data.tar.gz || echo "❗ Data extraction failed"
else
    echo "⚠️ data.tar.gz not found. Skipping extraction."
fi

echo "Files present:"
ls -l

# ==============================================================================
# ▶️ Run Python Script
# ==============================================================================
echo "=== STARTING MAIN SCRIPT ==="
"$PYTHON_EXEC" ./run_mlhc.py "$@"

echo "✅ Script completed at $(date)"
