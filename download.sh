wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2 -P ./Resource/
unzip enwiki_20180420_100d.txt.bz2 -d ./Resource/
rm ./Resource/enwiki_20180420_100d.txt.bz2

wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz -P ./Resource/
unzip cc.en.300.bin.gz -d ./Resource/
rm ./Resource/cc.en.300.bin.gz