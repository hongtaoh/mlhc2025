# Remove old tarball (clean start)
rm -rf env.tar.gz


# conda deactivate   # Exit any currently active env (safe practice)
# conda remove --name mlhc --all -y  # Clean old env entirely
# conda create -n mlhc python=3.9 -y  # New environment
# conda activate mlhc

# # Install conda-pack to make portable env
# conda install -c conda-forge conda-pack -y

# # Install pip-only packages
# pip install --no-cache-dir --force-reinstall pyebm pysaebm pyyaml
# pip install --no-cache-dir git+https://github.com/noxtoby/awkde
# pip install --no-cache-dir git+https://github.com/ucl-pond/kde_ebm


pip install --upgrade --no-cache-dir pysaebm 

# Verify installation
python -c "from kde_ebm import mixture_model; import kde_ebm, pyebm, pysaebm, scipy, yaml; import scipy._lib; print('✅ Dependencies OK')" 

# Package the environment
conda-pack -n mlhc --output env.tar.gz 