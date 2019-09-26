#!/usr/bin/env bash
labels=(airplane automobile bird cat deer dog frog horse ship truck)

for i in {1..9}; 
do
  somelabels=${labels[@]:0:${i}}
  ./extractClassFromCifar.sh rundirs/Cifar$i/source ${somelabels// /,} 5000
done
exit

i=5
somelabels=${labels[@]:0:${i}}
./extractClassFromCifar.sh rundirs/Cifar${i}/target ${somelabels// /,} 5000 test

#i=5
#somelabels=${labels[@]:0:${i}}
#./extractClassFromCifar.sh rundirs/ManyCifar${i}/target ${somelabels// /,} 25000 train 
#./extractClassFromCifar.sh rundirs/ManyCifar${i}/source ${somelabels// /,} 5000 test
#somelabels=${labels[@]:${i}:$((10-i))}
#./extractClassFromCifar.sh rundirs/ManyCifar${i}bis/source ${somelabels// /,} 25000 train
#./extractClassFromCifar.sh rundirs/ManyCifar${i}bis/target ${somelabels// /,} 5000 test
