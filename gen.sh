rm -rf data data.tar.gz ../data ../data.tar.gz
python3 gen.py
tar -czf data.tar.gz data
# mv data.tar.gz /staging/hhao9/
rm -rf data