#!/usr/bin/env bash

RUNDIR=${1-NONEXISTINGDIR}
LAUNCHDIR=`dirname $0`
LAUNCHDIR=`realpath $LAUNCHDIR`

for subdir in ${RUNDIR}/*;
do
  if [ -d $subdir ];
  then
    if [[ ! -d ${subdir}/results && -d ${subdir}/source && -d $subdir/target ]]; then
    #if [[ true ]]; then
      echo Dealing with ${subdir}
      mkdir ${subdir}/results
      label=`basename $subdir`
      python prd_from_image_folders.py --reference_dir $subdir/target --eval_dirs $subdir/source --eval_labels $label/cluster  --plot_path $subdir/results/cluster.png --num_clusters 20 | tail -n 1 | cut -d " " -f 1,2 > $subdir/results/cluster.score
      python prd_from_image_folders.py --reference_dir $subdir/target --eval_dirs $subdir/source --eval_labels $label/classif --plot_path $subdir/results/classif.png --classif --num_epochs=50 --patience=30 --num_runs 10 | tail -n 1 | cut -d " " -f 1,2 > $subdir/results/classif.score
    fi
  fi
done


