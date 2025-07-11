python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir --upgrade alabebm
tar -czf venv.tar.gz venv
# rm -rf venv
# deactivate