#!/bin/bash

in_dir=$1
out_dir=$2
runner=$3
echo "Processing $in_dir to $out_dir using $runner"

if [ ! -d $in_dir ]
then
  echo "$in_dir doesn't exist. Abort!"
  exit
fi

mkdir -p $out_dir

flist=$(ls $in_dir/*.ipynb)
status=0
for fsc in $flist 
do
  echo "  Processing $fsc"
  python3 $runner $fsc $out_dir html
  if [ $? -ne 0 ]; then
    status=1
  fi
done

echo "Done"

exit $status
