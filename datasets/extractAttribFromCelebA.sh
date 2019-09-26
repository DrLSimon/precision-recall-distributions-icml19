#!/usr/bin/env bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
  echo $0 outdir [label:value=gender:1] [nb=500] [startline=0]
  echo EXAMPLES:
  echo "   #extracting 250 first girls"
  echo "   $0 250girls gender:-1 250"
  echo "   #extracting 750 images after 26000"
  echo "   $0 750images all 750 26000"
  exit
fi

declare -A ATTRIBS=([gender]='22' [young]='41' [smile]='33')


DIR=$1
ATTRIB=${2-gender:1}
IFS=: read -r LABEL VALUE <<< "$ATTRIB"
NB=${3-500}
STARTLINE=${4-0}
FILTER=yes
if [[ $LABEL == all ]]; then
  FILTER=no
fi

if true
then
  LABELDIR=`dirname $0`
  LABELDIR=`realpath $LABELDIR`
  LABELFILE=${LABELDIR}/../../CelebAHQ/datasets/list_attr_celebahq.txt 

  IMGDIR=`dirname $0`
  IMGDIR=`realpath $IMGDIR`/celebahq_128
  format="%05d"
  baseindex=0
else
  IMGDIR=`dirname $0`
  IMGDIR=`realpath $IMGDIR`/celeba
  LABELFILE=${IMGDIR}/list_attr_celeba.txt 
  format="%06d"
  baseindex=1
fi



mkdir -p $DIR
cd $DIR
rm -f ./*
awk -v value=$VALUE -v sline=$STARTLINE -v nb=$NB -v attr=${ATTRIBS[$LABEL]} -v filter=$FILTER -v format=$format -v baseindex=$baseindex\
  '((filter == "no" || $attr == value) && cnt<nb && NR>sline && NR>2-baseindex){cnt++; print sprintf(format, NR-3+baseindex)".jpg"} ' \
  $LABELFILE | 
  xargs -I{} ln -s ${IMGDIR}/{} ./
