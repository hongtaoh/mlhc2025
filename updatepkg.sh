rm -rf env.tar.gz  # Remove old tarball (clean start)

conda activate mlhc

# conda deactivate   # Exit any currently active env (safe practice)

# Optional: clean old env entirely if needed
# conda remove --name mlhc --all -y

# conda create -n mlhc python=3.9 -y  # New environment
# conda activate mlhc

# # Install conda-pack to make portable env
# conda install -c conda-forge conda-pack -y

# # Install packages cleanly
# pip install --no-cache-dir --force-reinstall pyebm alabebm pyyaml
# pip install --no-cache-dir git+https://github.com/ucl-pond/kde_ebm 
pip install --upgrade --no-cache-dir pysaebm

# Verify installation is complete (esp. scipy._lib)
python -c "import kde_ebm, pyebm, pysaebm, scipy; import yaml; import scipy._lib; print('✅ Dependencies OK')"

# Package it
conda-pack -n mlhc --output env.tar.gz

