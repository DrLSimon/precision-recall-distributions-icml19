#!/usr/bin/env bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
  echo $0 outdir [label:value=all] [nb=500] [set=train]
  echo EXAMPLES:
  echo "   #extracting 250 first girls"
  echo "   $0 250girls gender:-1 250"
  echo "   #extracting 750 images after 26000"
  echo "   $0 750images all 750 26000"
  exit
fi



DIR=$1
LABELS=${2-all}
NB=${3-500}
SET=${4-train}
SKIP=${5-0}
FILTER=yes
if [[ $LABELS == all ]]; then
  FILTER=no
fi

IMGDIR=`dirname $0`
IMGDIR=`realpath $IMGDIR`/cifar/$SET/


mkdir -p $DIR
cd $DIR
rm -f ./*
cnt=0
skipped=0
allfiles=`ls $IMGDIR/ | sort -V `
for f in $allfiles; 
do
  if [[ ! $cnt -lt $NB ]]; then
    exit
  fi
  for LABEL in $(echo $LABELS | sed "s/,/ /g"); do 
    if [[ ($FILTER == no) || ($f == *"${LABEL}.png") ]]; then
      if [[ $skipped -lt $SKIP ]]; then
        skipped=$((skipped+1))
      else
        ln -s $IMGDIR/$f ./
        cnt=$((cnt+1))
      fi
    fi
  done
done
