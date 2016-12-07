#!/bin/bash
while read p; do
  pip install $p
done < externals_links.txt
