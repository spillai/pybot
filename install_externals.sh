#!/bin/bash
mkdir -p tmp_dir; 
cd tmp_dir
while read p; do
    wget $p
done < ../externals_links.txt
pip install *.tar.gz
conda install *.bz2
cd ../
rm -Rf tmp_dir
