#!/bin/sh
# if ! type pip > /dev/null; then
#     echo "Installing pip"
#     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#     python get-pip.py
# fi

# echo "Installing venv"
# pip3 install --upgrade pip
# pip3 install virtualenv

# apt-get install python3-venv

# echo "Creating Virtual Env"
# python -m venv .env
# source .env/bin/activate

echo "Installing Basic Conda Packages"
pip install -r requirements.txt

echo "Installing Gensim"
pip install --upgrade gensim

echo "Installing Torch"
pip install torch torchvision

echo "Installing Spacy"
pip install -U spacy

echo "Downloading Spacy Corpus"
python -m spacy download en_core_web_sm

echo "Building neuralcoref from source"
cd ..
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e .
cd ..
cd GIM_ASAG

echo "Downloading word2vec vectors"
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2 -P ./Resource/
bzip2 enwiki_20180420_100d.txt.bz2 -d ./Resource/
rm ./Resource/enwiki_20180420_100d.txt.bz2

echo "Downloading fastText vectors"
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P ./Resource/
gunzip cc.en.300.bin.gz -d ./Resource/
rm ./Resource/cc.en.300.bin.gz