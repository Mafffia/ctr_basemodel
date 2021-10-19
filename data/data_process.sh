#! /bin/bash
# This will download the data and preprocess it.

set -e
echo "begin download data"
mkdir -p raw_data # create the folder `raw_data` if it does not already exist
cd raw_data
wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
echo "download data successfully"

cd ..
python data_preprocess.py