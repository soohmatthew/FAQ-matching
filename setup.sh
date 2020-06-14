#!/bin/sh
echo "Update Packages"
apt-get update -y
apt-get upgrade -y
apt install python3-pip -y
apt install python3-venv -y

echo "Upgrading Pip"
pip3 install --upgrade pip -y

cd ..
echo "Creating Virtual Env"
python3 -m venv .asag_env
source .asag_env/bin/activate
cd gim_asag

echo "Installing Basic Conda Packages"
pip3 install -r requirements.txt

echo "Installing Spacy"
pip3 install -U spacy

echo "Downloading Spacy Corpus"
python3 -m spacy download en_core_web_sm

echo "Building neuralcoref from source"
cd ..
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e .
cd ..
cd gim_asag

echo "Downloading word2vec vectors"
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2 -P ./Resource/
bzip2 -d ./Resource/enwiki_20180420_100d.txt.bz2

echo "Downloading fastText vectors"
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P ./Resource/
gunzip -d ./Resource/cc.en.300.bin.gz